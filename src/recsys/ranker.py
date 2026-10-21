"""LightGBM LambdaRank re-ranker over the baseline candidates.

Train on val-period positives (features as of end of train), early-stop on held-out val users,
evaluate on the test split. Usage: python -m recsys.ranker [--limit N]
"""

import argparse
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl

from recsys.data import DB_PATH
from recsys.features import FEATURES, build_dataset, group_sizes
from recsys.metrics import bootstrap_ci, coverage, paired_bootstrap_diff, per_user_metrics
from recsys.split import eval_positives

KS = (10, 20)
CACHE = Path("data/cache")
RESULTS = Path(__file__).resolve().parents[2] / "docs" / "02-results.md"
MARKER = "## Ranker (M4)"
KNN_FEATURES = ["knn_score", "knn_rank"]
PARAMS = {
    "objective": "lambdarank", "learning_rate": 0.05, "n_estimators": 600, "num_leaves": 31,
    "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "lambdarank_truncation_level": 20, "random_state": 0, "verbose": -1,
}  # fmt: skip


def dataset(con, split: str, users: list[int]) -> pl.DataFrame:
    """Candidate features + labels for `users`, cached because the build takes minutes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}_{len(users)}_{users[0]}_{users[-1]}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    df = build_dataset(con, users, split)
    df.write_parquet(path)
    return df


def eval_users(con, split: str, limit: int | None) -> tuple[list[int], dict[int, set[int]]]:
    pos = eval_positives(con, split)
    truth = {u: set(g["movie_id"]) for (u,), g in pos.group_by("user_id")}
    users = sorted(truth)[:limit]
    return users, {u: truth[u] for u in users}


def fit(train: pl.DataFrame, es: pl.DataFrame, features: list[str]) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(**PARAMS)
    model.fit(
        train.select(features).to_numpy(), train["label"].to_numpy(), group=group_sizes(train),
        eval_set=[(es.select(features).to_numpy(), es["label"].to_numpy())],
        eval_group=[group_sizes(es)], eval_at=list(KS),
        callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)],
    )  # fmt: skip
    return model


def top_k(df: pl.DataFrame, score: np.ndarray, k: int) -> dict[int, np.ndarray]:
    ranked = df.select("user_id", "movie_id").with_columns(pl.Series("score", score))
    ranked = ranked.sort(["user_id", "score"], descending=[False, True])
    return {
        u: g["movie_id"].head(k).to_numpy() for (u,), g in ranked.group_by("user_id")
    }


def source_top_k(df: pl.DataFrame, col: str, k: int) -> dict[int, np.ndarray]:
    sub = df.filter(pl.col(col) <= k).sort("user_id", col)
    return {u: g["movie_id"].to_numpy() for (u,), g in sub.group_by("user_id")}


def drop_empty_groups(df: pl.DataFrame) -> pl.DataFrame:
    """Users with no positive among the candidates carry no ranking signal."""
    keep = df.group_by("user_id").agg(pl.col("label").sum()).filter(pl.col("label") > 0)
    return df.join(keep.select("user_id"), on="user_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="smoke test on the first N users per split")
    a = ap.parse_args()
    con = duckdb.connect(str(DB_PATH), read_only=True)

    val_users, val_truth = eval_users(con, "val", a.limit)
    test_users, test_truth = eval_users(con, "test", a.limit)
    val = dataset(con, "val", val_users)
    test = dataset(con, "test", test_users)
    rng = np.random.default_rng(0)
    es_users = set(rng.choice(val_users, size=max(1, len(val_users) // 5), replace=False).tolist())
    in_es = pl.col("user_id").is_in(list(es_users))
    train, es = drop_empty_groups(val.filter(~in_es)), drop_empty_groups(val.filter(in_es))
    print(f"train {train.height:,} rows / es {es.height:,} / test {test.height:,}", flush=True)

    n_items = con.execute(
        "SELECT count(DISTINCT movie_id) FROM split_ratings WHERE split='train' AND rating >= 4"
    ).fetchone()[0]
    results: dict[str, dict] = {}
    mlflow.set_experiment("ranker")
    for name, feats in {
        "LightGBM ranker": FEATURES,
        "  - without kNN score/rank": [f for f in FEATURES if f not in KNN_FEATURES],
    }.items():
        model = fit(train, es, feats)
        scores = model.predict(test.select(feats).to_numpy())
        recs = top_k(test, scores, max(KS))
        m = per_user_metrics(recs, test_truth, KS)
        es_truth = {u: val_truth[u] for u in es["user_id"].unique().to_list()}
        es_m = per_user_metrics(top_k(es, model.predict(es.select(feats).to_numpy()), 20),
                                es_truth, KS)  # fmt: skip
        with mlflow.start_run(run_name=name.strip()):
            mlflow.log_params({**PARAMS, "n_features": len(feats), "best_iter": model.best_iteration_})
            mlflow.log_metric("val_ndcg_at_10", float(es_m["ndcg@10"].mean()))
            mlflow.log_metric("val_recall_at_20", float(es_m["recall@20"].mean()))
            for k, v in m.items():
                mlflow.log_metric(k.replace("@", "_at_"), float(v.mean()))
        results[name] = {"m": m, "recs": recs, "model": model, "feats": feats,
                         "val_ndcg10": float(es_m["ndcg@10"].mean())}  # fmt: skip
        print(name, "val ndcg@10", round(results[name]["val_ndcg10"], 4),
              "test recall@20", round(float(m["recall@20"].mean()), 4), flush=True)  # fmt: skip

    # baselines on the same users, derived from the candidate ranks (same top-K as baselines.py)
    base = {}
    for label, col in (("Recent popularity (90d)", "recent_rank"), ("Item-kNN (cosine)", "knn_rank")):
        recs = source_top_k(test, col, 20)
        base[label] = {"m": per_user_metrics(recs, test_truth, KS), "recs": recs}
    ceiling = np.mean([
        len(test_truth[u] & set(g["movie_id"])) / len(test_truth[u])
        for (u,), g in test.group_by("user_id")
    ])  # fmt: skip
    if a.limit:
        return print("smoke run: results not written")
    write_results(results, base, ceiling, n_items, test.height, len(test_users))


def fmt(ci: tuple[float, float, float]) -> str:
    return f"{ci[0]:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]"


def write_results(results, base, ceiling, n_items, n_rows, n_users) -> None:
    rows = {**base, **results}
    lines = ["| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MAP@20 | coverage@20 |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, r in rows.items():
        m = r["m"]
        lines.append(
            f"| {name.strip()} | " + " | ".join(
                fmt(bootstrap_ci(m[k])) for k in
                ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "map@20")
            ) + f" | {coverage(r['recs'], n_items, 20):.3f} |"
        )
    full, abl = results["LightGBM ranker"], results["  - without kNN score/rank"]
    diffs = []
    for label, other in (("recent popularity", base["Recent popularity (90d)"]),
                         ("item-kNN", base["Item-kNN (cosine)"]),
                         ("ranker without kNN", abl)):  # fmt: skip
        for k in ("recall@20", "ndcg@10"):
            ci = paired_bootstrap_diff(full["m"][k], other["m"][k])
            diffs.append(f"- ranker − {label}, {k}: {fmt(ci)}")
    imp = sorted(zip(full["feats"], full["model"].booster_.feature_importance("gain")),
                 key=lambda t: -t[1])  # fmt: skip
    tot = sum(g for _, g in imp)
    imp_lines = [f"| {f} | {g / tot:.1%} |" for f, g in imp]
    section = (
        f"{MARKER}\n\nRe-ranks the union of the top-100 from popularity, recent popularity and "
        f"item-kNN ({n_rows:,} candidate rows for {n_users:,} test users). Candidate recall "
        f"ceiling (share of a user's test positives inside the candidate set, averaged over "
        f"users): {ceiling:.4f}. Trained on val-period positives with features as of the end of "
        "train; early-stopped on 20% of val users. Baseline rows here are recomputed on the same "
        "users and match the table above.\n\n" + "\n".join(lines)
        + "\n\nPaired bootstrap differences (same users resampled; mean [95% CI]):\n\n"
        + "\n".join(diffs)
        + "\n\nFeature importance (gain share, full model):\n\n| feature | share |\n|---|---|\n"
        + "\n".join(imp_lines) + "\n"
    )  # fmt: skip
    text = RESULTS.read_text() if RESULTS.exists() else "# Results\n"
    head, _, rest = text.partition(MARKER)
    later = rest.partition("\n## ")[2]  # sections written after this one (e.g. M5) are kept
    RESULTS.write_text(head.rstrip() + "\n\n" + section + (f"\n## {later}" if later else ""))


if __name__ == "__main__":
    main()

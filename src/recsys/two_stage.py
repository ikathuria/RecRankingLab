"""Milestone 5: two-tower retrieval vs the baselines, and the two-stage system.

1. Train the two-tower on train positives, selecting the epoch by Recall@100 on val users.
2. Retrieval: Recall@100/200 on test users against popularity, recent popularity and item-kNN.
3. Two-stage: the two-tower's top-200 per user, re-ranked by LightGBM LambdaRank (same protocol
   as ranker.py: features as of train end, trained on val-period positives), against the
   single-stage ranker over the baseline candidates.
Both rankers and the retriever are logged and registered in MLflow.

torch and lightgbm bundle different libomp builds; training LightGBM in a process that has
loaded torch segfaults on macOS. So the work is two processes sharing a pickle: `retrieve`
(torch) and `rank` (lightgbm, never imports torch). Usage:
python -m recsys.two_stage [retrieve|rank|all] [--limit N] [--epochs E]
"""
import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

if "rank" not in sys.argv:  # every stage but `rank` may use torch, which must load first
    import torch  # noqa: F401

import duckdb
import mlflow
import numpy as np
import polars as pl

from recsys import ranker
from recsys.baselines import item_knn, popularity, recent_popularity
from recsys.data import DB_PATH
from recsys.features import FEATURES, build_dataset
from recsys.metrics import bootstrap_ci, paired_bootstrap_diff, per_user_metrics

RETRIEVAL_KS = (100, 200)
N_RETRIEVE = 200
TT_FEATURES = [*FEATURES, "tt_score", "tt_rank"]
STATE = ranker.CACHE / "two_stage_{limit}.pkl"
TT = "Two-tower (exact dot product)"
MARKER = "## Two-tower retrieval and two-stage ranking (M5)"


def fmt(ci: tuple[float, float, float]) -> str:
    return f"{ci[0]:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]"


def cached_dataset(con, split, users, retrieved) -> pl.DataFrame:
    ranker.CACHE.mkdir(parents=True, exist_ok=True)
    path = ranker.CACHE / f"tt{N_RETRIEVE}_{split}_{len(users)}_{users[0]}_{users[-1]}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    df = build_dataset(con, users, split, retrieved=retrieved)
    df.write_parquet(path)
    return df


def split_es(val: pl.DataFrame, val_users: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Same 20% early-stopping user split as ranker.main."""
    rng = np.random.default_rng(0)
    es_users = rng.choice(val_users, size=max(1, len(val_users) // 5), replace=False).tolist()
    in_es = pl.col("user_id").is_in(es_users)
    return ranker.drop_empty_groups(val.filter(~in_es)), ranker.drop_empty_groups(val.filter(in_es))


def recall_at(recs, truth, ks) -> dict[str, np.ndarray]:
    return per_user_metrics(recs, truth, ks)


def state_path(limit: int | None) -> Path:
    return Path(str(STATE).format(limit=limit or "all"))


def retrieve_stage(a) -> None:
    """Train the two-tower and retrieve for val and test users (torch process)."""
    from recsys import twotower

    con = duckdb.connect(str(DB_PATH), read_only=True)
    val_users, val_truth = ranker.eval_users(con, "val", a.limit)
    test_users, test_truth = ranker.eval_users(con, "test", a.limit)
    data = twotower.prepare(con)
    print(f"{data.n_users:,} users, {data.n_items:,} items, {len(data.pairs):,} pairs", flush=True)

    def on_epoch(model, epoch):
        recs = twotower.retrieve(model, data, con, val_users, k=100)
        r = per_user_metrics({u: v[0] for u, v in recs.items()}, val_truth, (100,))["recall@100"]
        return {"val_recall_100": float(r.mean()), "select": float(r.mean())}

    config = {"dim": 64, "hidden": 128, "epochs": a.epochs, "batch": 4096, "lr": 2e-3, "seed": 0}
    t0 = time.time()
    model, history = twotower.train(data, **config, on_epoch=on_epoch)
    train_secs = time.time() - t0
    best_epoch = max(history, key=lambda h: h["val_recall_100"])["epoch"]

    t0 = time.time()
    tt_test = twotower.retrieve(model, data, con, test_users, k=N_RETRIEVE)
    tt_secs = time.time() - t0
    tt_val = twotower.retrieve(model, data, con, val_users, k=N_RETRIEVE)
    retrieval = {TT: {u: v[0] for u, v in tt_test.items()}}
    for name, fn in (("Popularity", popularity), ("Recent popularity (90d)", recent_popularity),
                     ("Item-kNN (cosine)", item_knn)):  # fmt: skip
        retrieval[name] = fn(con, test_users, k=N_RETRIEVE)
    ret_m = {n: per_user_metrics(r, test_truth, RETRIEVAL_KS) for n, r in retrieval.items()}
    for n, m in ret_m.items():
        print(f"{n}: recall@100 {m['recall@100'].mean():.4f}  recall@200 {m['recall@200'].mean():.4f}")

    state = {"tt_test": tt_test, "tt_val": tt_val, "ret_m": ret_m, "history": history,
             "best_epoch": best_epoch, "train_secs": train_secs, "tt_secs": tt_secs,
             "n_users": data.n_users, "n_items": data.n_items, "n_pairs": len(data.pairs),
             "temperature": twotower.TEMPERATURE}  # fmt: skip
    ranker.CACHE.mkdir(parents=True, exist_ok=True)
    state_path(a.limit).write_bytes(pickle.dumps(state))
    if not a.limit:
        mlflow.set_experiment("two-stage")
        with mlflow.start_run(run_name="two-tower"):
            mlflow.log_params({**config, "best_epoch": best_epoch,
                               "temperature": twotower.TEMPERATURE})  # fmt: skip
            for h in history:
                mlflow.log_metrics({"loss": h["loss"], "val_recall_at_100": h["val_recall_100"]},
                                   step=h["epoch"])  # fmt: skip
            mlflow.log_metric("train_seconds", train_secs)
            for k in RETRIEVAL_KS:
                mlflow.log_metric(f"test_recall_at_{k}", float(ret_m[TT][f"recall@{k}"].mean()))
            mlflow.pytorch.log_model(model, name="model", registered_model_name="two-tower-retriever",
                                     serialization_format="pickle")


def rank_stage(a) -> None:
    """Re-rank the two-tower's candidates and the baseline candidates (lightgbm process)."""
    st = pickle.loads(state_path(a.limit).read_bytes())
    con = duckdb.connect(str(DB_PATH), read_only=True)
    val_users, _ = ranker.eval_users(con, "val", a.limit)
    test_users, test_truth = ranker.eval_users(con, "test", a.limit)
    val2 = cached_dataset(con, "val", val_users, st["tt_val"])
    test2 = cached_dataset(con, "test", test_users, st["tt_test"])
    val1 = ranker.dataset(con, "val", val_users)
    test1 = ranker.dataset(con, "test", test_users)
    systems = {}
    for name, (val, test, feats) in {
        "Single-stage LightGBM (baseline candidates)": (val1, test1, FEATURES),
        "Two-stage: two-tower top-200 -> LightGBM": (val2, test2, TT_FEATURES),
    }.items():
        train, es = split_es(val, val_users)
        gbm = ranker.fit(train, es, feats)
        recs = ranker.top_k(test, gbm.predict(test.select(feats).to_numpy()), 20)
        ceiling = float(np.mean([len(test_truth[u] & set(g["movie_id"])) / len(test_truth[u])
                                 for (u,), g in test.group_by("user_id")]))  # fmt: skip
        systems[name] = {"m": per_user_metrics(recs, test_truth, ranker.KS), "gbm": gbm,
                         "ceiling": ceiling, "best_iter": gbm.best_iteration_}  # fmt: skip
        print(name, "ndcg@10", round(float(systems[name]["m"]["ndcg@10"].mean()), 4), flush=True)
    tt_alone = per_user_metrics({u: v[0][:20] for u, v in st["tt_test"].items()}, test_truth,
                                ranker.KS)  # fmt: skip
    if a.limit:
        return print("smoke run: nothing logged or written")
    mlflow.set_experiment("two-stage")
    for name, key in (("Single-stage", "single-stage-ranker"), ("Two-stage", "two-stage-ranker")):
        s = next(v for n, v in systems.items() if n.startswith(name))
        with mlflow.start_run(run_name=key):
            mlflow.log_params({**ranker.PARAMS, "best_iter": s["best_iter"]})
            for k, v in s["m"].items():
                mlflow.log_metric(f"test_{k.replace('@', '_at_')}", float(v.mean()))
            mlflow.lightgbm.log_model(s["gbm"].booster_, name="model", registered_model_name=key)
    write_results(st, systems, tt_alone, len(test_users), len(val_users))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["retrieve", "rank", "all"])
    ap.add_argument("--limit", type=int, help="smoke test on the first N users per split")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "retrieve":
        retrieve_stage(a)
    elif a.stage == "rank":
        rank_stage(a)
    else:
        extra = ["--limit", str(a.limit)] if a.limit else []
        for stage in ("retrieve", "rank"):
            cmd = [sys.executable, "-m", "recsys.two_stage", stage, "--epochs", str(a.epochs), *extra]
            subprocess.run(cmd, check=True)


def write_results(st, systems, tt_alone, n_test, n_val) -> None:
    ret_m, history, best_epoch = st["ret_m"], st["history"], st["best_epoch"]
    train_secs, tt_secs = st["train_secs"], st["tt_secs"]
    ret = ["| retriever | Recall@100 | Recall@200 |", "|---|---|---|"]
    for n, m in ret_m.items():
        ret.append(f"| {n} | {fmt(bootstrap_ci(m['recall@100']))} | {fmt(bootstrap_ci(m['recall@200']))} |")  # fmt: skip
    tt = ret_m[TT]
    diffs = []
    for n, m in ret_m.items():
        if n != TT:
            for k in RETRIEVAL_KS:
                diffs.append(f"- two-tower − {n}, recall@{k}: "
                             f"{fmt(paired_bootstrap_diff(tt[f'recall@{k}'], m[f'recall@{k}']))}")  # fmt: skip
    hist = ["| epoch | train loss | val Recall@100 |", "|---|---|---|"]
    hist += [f"| {h['epoch']} | {h['loss']:.3f} | {h['val_recall_100']:.4f} |" for h in history]
    single, two = list(systems.values())
    end = ["| system | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | candidate recall ceiling |",
           "|---|---|---|---|---|---|"]  # fmt: skip
    for n, s in [("Two-tower alone (top-20 by dot product)", {"m": tt_alone, "ceiling": None}),
                 *systems.items()]:  # fmt: skip
        c = "" if s["ceiling"] is None else f"{s['ceiling']:.4f}"
        end.append(f"| {n} | " + " | ".join(fmt(bootstrap_ci(s["m"][k])) for k in
                   ("recall@10", "recall@20", "ndcg@10", "ndcg@20")) + f" | {c} |")  # fmt: skip
    sd = [f"- two-stage − single-stage, {k}: {fmt(paired_bootstrap_diff(two['m'][k], single['m'][k]))}"
          for k in ("ndcg@10", "ndcg@20", "recall@10", "recall@20")]  # fmt: skip
    section = (
        f"{MARKER}\n\nTwo-tower: user-id and item-id(+genre) towers, 64-d L2-normalised "
        f"embeddings, in-batch sampled softmax with logQ correction, temperature "
        f"{st['temperature']}, batch 4096, Adam 2e-3, {st['n_users']:,} users x "
        f"{st['n_items']:,} items, {st['n_pairs']:,} train positives; trained in "
        f"{train_secs / 60:.1f} min on CPU, epoch {best_epoch} kept (best val Recall@100 on "
        f"{n_val:,} val users). Retrieval over all items is an exact dot product "
        f"({tt_secs:.0f} s for {n_test:,} test users). Items the user rated in train are excluded "
        "for every retriever.\n\n### Retrieval on the test split\n\n"
        + "\n".join(ret) + "\n\nPaired bootstrap differences (mean [95% CI]):\n\n" + "\n".join(diffs)
        + "\n\nTraining history:\n\n" + "\n".join(hist)
        + f"\n\n### Two-stage vs single-stage ({n_test:,} test users)\n\n"
        "Both rankers use the same features (the two-stage one adds tt_score and tt_rank), "
        "LightGBM parameters and training protocol as M4. The candidate recall ceiling is the "
        "share of a user's test positives inside the candidate set.\n\n"
        + "\n".join(end) + "\n\nPaired bootstrap differences:\n\n" + "\n".join(sd) + "\n"
    )  # fmt: skip
    text = ranker.RESULTS.read_text()
    head, _, _ = text.partition(MARKER)
    ranker.RESULTS.write_text(head.rstrip() + "\n\n" + section)


if __name__ == "__main__":
    main()

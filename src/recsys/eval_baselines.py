"""Evaluate the baselines on a split, log to MLflow, write the table to docs/02-results.md.

Usage: python -m recsys.eval_baselines [--split test] [--limit N] [--no-write]
"""

import argparse
import time
from pathlib import Path

import duckdb
import mlflow

from recsys.baselines import item_knn, popularity, recent_popularity
from recsys.data import DB_PATH
from recsys.metrics import bootstrap_ci, coverage, per_user_metrics
from recsys.split import eval_positives

KS = (10, 20)
RESULTS = Path(__file__).resolve().parents[2] / "docs" / "02-results.md"
MODELS = {
    "Popularity": popularity,
    "Recent popularity (90d)": recent_popularity,
    "Item-kNN (cosine)": item_knn,
}


def run(split: str, limit: int | None) -> tuple[list[dict], dict]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pos = eval_positives(con, split)
    truth = {u: set(g["movie_id"]) for (u,), g in pos.group_by("user_id")}
    users = sorted(truth)[:limit]
    truth = {u: truth[u] for u in users}
    n_items = con.execute(
        "SELECT count(DISTINCT movie_id) FROM split_ratings WHERE split='train' AND rating >= 4"
    ).fetchone()[0]
    rows = []
    for name, fn in MODELS.items():
        t0 = time.time()
        recs = fn(con, users, k=max(KS))
        secs = time.time() - t0
        m = per_user_metrics(recs, truth, KS)
        row = {"model": name, "seconds": secs, "users": len(users)}
        for metric, vals in m.items():
            row[metric] = bootstrap_ci(vals)
        for k in KS:
            row[f"coverage@{k}"] = coverage(recs, n_items, k)
        rows.append(row)
        print(f"{name}: recall@20={row['recall@20'][0]:.4f} ({secs:.0f}s)", flush=True)
    return rows, {"n_items": n_items, "positives": sum(len(v) for v in truth.values())}


def table(rows: list[dict]) -> str:
    def f(ci):
        return f"{ci[0]:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]"

    head = "| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MAP@20 | coverage@20 |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['model']} | {f(r['recall@10'])} | {f(r['recall@20'])} | {f(r['ndcg@10'])} | "
            f"{f(r['ndcg@20'])} | {f(r['map@20'])} | {r['coverage@20']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rows, meta = run(a.split, a.limit)
    mlflow.set_experiment("baselines")
    for r in rows:
        with mlflow.start_run(run_name=f"{r['model']} / {a.split}"):
            mlflow.log_params({"model": r["model"], "split": a.split, "users": r["users"]})
            mlflow.log_metric("seconds", r["seconds"])
            for key, v in r.items():
                if isinstance(v, tuple):
                    n = key.replace("@", "_at_")
                    mlflow.log_metrics({n: v[0], f"{n}_lo": v[1], f"{n}_hi": v[2]})
                elif key.startswith("coverage"):
                    mlflow.log_metric(key.replace("@", "_at_"), v)
    md = table(rows)
    print(md)
    if a.no_write or a.limit:
        return
    RESULTS.write_text(
        f"# Results\n\n## Baselines on the {a.split} split (M3)\n\n"
        f"{rows[0]['users']:,} eligible users with a positive, {meta['positives']:,} positives. "
        "Cells are mean [95% bootstrap CI over users, 1000 resamples, seed 0]. Coverage is the "
        f"share of the {meta['n_items']:,} train-positive items recommended to anyone.\n\n{md}\n\n"
        "Recall is capped below 1 because 12.8% of test ratings are on items unseen in train "
        "(docs/01-evaluation-protocol.md). Candidates: train items minus the user's train items.\n"
    )


if __name__ == "__main__":
    main()

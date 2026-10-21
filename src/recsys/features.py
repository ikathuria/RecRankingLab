"""SQL feature pipeline, candidate generation and ranker training sets.

Every feature table is computed from ratings with ts <= as_of (sql/*.sql take `$as_of`). The
ranker's features are always as of the end of train, so train, val and test rows are comparable
and the baselines and the ranker see the same history (docs/01-evaluation-protocol.md).
"""

from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from recsys.baselines import (
    KnnScorer,
    _popularity_order,
    _seen,
    _top_k_by_rank,
    _train_positives,
)
from recsys.split import eval_positives

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
RECENT_DAYS = 90
SOURCE_RANKS = ["pop_rank", "recent_rank", "knn_rank"]
FEATURES = [
    "knn_score", *SOURCE_RANKS, "genre_match", "item_vs_user_mean",
    "i_n_ratings", "i_n_pos", "i_mean_rating", "i_pos_rate", "i_pos_90d", "i_age_years",
    "i_n_genres", "u_n_ratings", "u_mean_rating", "u_pos_rate", "u_days_since_last",
    "u_span_days",
]  # fmt: skip


def train_end(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute("SELECT max(ts) FROM split_ratings WHERE split = 'train'").fetchone()[0]


def build_feature_tables(con: duckdb.DuckDBPyConnection, as_of: int) -> None:
    """Create temp tables user_features, item_features, user_genre as of `as_of`."""
    for name in ("user_features", "item_features", "user_genre"):
        con.execute((SQL_DIR / f"{name}.sql").read_text(), {"as_of": as_of})


def build_candidates(
    con: duckdb.DuckDBPyConnection, users: list[int], n: int = 100
) -> pl.DataFrame:
    """Union of each user's top-n from popularity, recent popularity and item-kNN.

    Columns: user_id, movie_id, knn_score, pop_rank, recent_rank, knn_rank (1-based rank in that
    source, null when the item is not in that source's top-n).
    """
    pos = _train_positives(con)
    pop = _popularity_order(pos)
    recent = _popularity_order(pos.filter(pl.col("ts") > pos["ts"].max() - RECENT_DAYS * 86400))
    seen = _seen(con, users)
    model = KnnScorer(con)

    cands: dict[int, dict[int, dict[str, float]]] = {u: {} for u in users}
    for u in users:
        for col, order in (("pop_rank", pop), ("recent_rank", recent)):
            for r, m in enumerate(_top_k_by_rank(order, seen.get(u, set()), n), 1):
                cands[u].setdefault(int(m), {})[col] = r
    for chunk, scores in model.batches(con, users):
        for r, u in enumerate(chunk):
            top = np.argpartition(-scores[r], min(n, scores.shape[1] - 1))[:n]
            top = top[np.argsort(-scores[r, top], kind="stable")]
            top = top[np.isfinite(scores[r, top])]  # drop already-rated items
            for rank, c in enumerate(top, 1):
                cands[u].setdefault(int(model.items[c]), {})["knn_rank"] = rank
            # kNN score for every candidate, including those that came from other sources
            for m, d in cands[u].items():
                c = model.item_col.get(m)
                d["knn_score"] = float(scores[r, c]) if c is not None else 0.0
    rows = [
        {"user_id": u, "movie_id": m, "knn_score": d.get("knn_score", 0.0),
         **{c: d.get(c) for c in SOURCE_RANKS}}
        for u, items in cands.items() for m, d in items.items()
    ]  # fmt: skip
    schema = {"user_id": pl.Int64, "movie_id": pl.Int64, "knn_score": pl.Float64} | {
        c: pl.Int64 for c in SOURCE_RANKS
    }
    return pl.DataFrame(rows, schema=schema)


def pair_features(con: duckdb.DuckDBPyConnection, cand: pl.DataFrame, as_of: int) -> pl.DataFrame:
    """Candidate rows joined with item, user and user-item features as of `as_of`."""
    build_feature_tables(con, as_of)
    con.register("cand_df", cand.to_arrow())
    con.execute("CREATE OR REPLACE TEMP TABLE cand AS SELECT * FROM cand_df")
    con.unregister("cand_df")
    con.execute((SQL_DIR / "pair_features.sql").read_text())
    return con.execute("SELECT * FROM pair_features ORDER BY user_id, movie_id").pl()


def build_dataset(
    con: duckdb.DuckDBPyConnection, users: list[int], label_split: str, n: int = 100
) -> pl.DataFrame:
    """Features at the end of train for each candidate of `users`, labelled with whether the
    item is a positive in `label_split`. Rows are sorted by user_id so groups are contiguous."""
    cand = build_candidates(con, users, n)
    df = pair_features(con, cand, train_end(con))
    pos = eval_positives(con, label_split).with_columns(pl.lit(1).alias("label"))
    return (
        df.join(pos, on=["user_id", "movie_id"], how="left")
        .with_columns(pl.col("label").fill_null(0))
        .sort("user_id", "movie_id")
    )


def group_sizes(df: pl.DataFrame) -> list[int]:
    """Rows per user for LightGBM's `group` (df must be sorted by user_id)."""
    return df.group_by("user_id", maintain_order=True).len()["len"].to_list()

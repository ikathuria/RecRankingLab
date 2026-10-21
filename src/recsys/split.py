"""Leakage-safe global time split. See docs/01-evaluation-protocol.md."""

from dataclasses import dataclass

import duckdb

POSITIVE_RATING = 4.0
MIN_HISTORY = 20  # train ratings a user needs to be evaluated


@dataclass(frozen=True)
class SplitInfo:
    train_end: int  # unix seconds; train is ts <= train_end
    val_end: int  # val is train_end < ts <= val_end; test is ts > val_end
    eligible_users: int
    cold_start_users: int  # users with a val/test positive but too little history


def make_split(
    con: duckdb.DuckDBPyConnection,
    train_q: float = 0.8,
    val_q: float = 0.9,
    min_history: int = MIN_HISTORY,
) -> SplitInfo:
    """Create `split_ratings` (all ratings tagged train/val/test) and `eligible_users`."""
    train_end, val_end = con.execute(
        "SELECT quantile_disc(ts, ?), quantile_disc(ts, ?) FROM ratings", [train_q, val_q]
    ).fetchone()
    con.execute(
        """
        CREATE OR REPLACE TABLE split_ratings AS
        SELECT user_id, movie_id, rating, ts,
               CASE WHEN ts <= ? THEN 'train' WHEN ts <= ? THEN 'val' ELSE 'test' END AS split
        FROM ratings
        """,
        [train_end, val_end],
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE eligible_users AS
        SELECT user_id FROM split_ratings WHERE split = 'train'
        GROUP BY user_id HAVING count(*) >= ?
        """,
        [min_history],
    )
    eligible = con.execute("SELECT count(*) FROM eligible_users").fetchone()[0]
    cold = con.execute(
        """
        SELECT count(DISTINCT user_id) FROM split_ratings
        WHERE split <> 'train' AND rating >= ?
          AND user_id NOT IN (SELECT user_id FROM eligible_users)
        """,
        [POSITIVE_RATING],
    ).fetchone()[0]
    return SplitInfo(int(train_end), int(val_end), eligible, cold)


def eval_positives(con: duckdb.DuckDBPyConnection, split: str):
    """(user_id, movie_id) positives of eligible users in `split` ('val' or 'test')."""
    if split not in ("val", "test"):
        raise ValueError(split)
    return con.execute(
        """
        SELECT user_id, movie_id FROM split_ratings
        WHERE split = ? AND rating >= ? AND user_id IN (SELECT user_id FROM eligible_users)
        """,
        [split, POSITIVE_RATING],
    ).pl()

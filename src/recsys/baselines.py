"""Non-learned baselines. Each returns {user_id: top-K item ids} ranked best first.

Candidates are items with a positive in train, minus everything the user rated in train
(docs/01-evaluation-protocol.md). Only train data is read.
"""

import duckdb
import numpy as np
import polars as pl
import scipy.sparse as sp

from recsys.split import POSITIVE_RATING

RECENT_DAYS = 90


def _train_positives(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return con.execute(
        "SELECT user_id, movie_id, ts FROM split_ratings WHERE split = 'train' AND rating >= ?",
        [POSITIVE_RATING],
    ).pl()


def _seen(con: duckdb.DuckDBPyConnection, users: list[int]) -> dict[int, set[int]]:
    df = con.execute(
        "SELECT user_id, list(movie_id) AS items FROM split_ratings "
        "WHERE split = 'train' AND user_id IN (SELECT unnest(?)) GROUP BY user_id",
        [users],
    ).pl()
    return {u: set(items) for u, items in zip(df["user_id"], df["items"])}


def _top_k_by_rank(order: np.ndarray, seen: set[int], k: int) -> np.ndarray:
    out = [i for i in order if i not in seen]
    return np.asarray(out[:k]) if len(out) >= k else np.asarray(out)


def _popularity_order(pos: pl.DataFrame) -> np.ndarray:
    counts = pos.group_by("movie_id").len().sort(["len", "movie_id"], descending=[True, False])
    return counts["movie_id"].to_numpy()


def popularity(con: duckdb.DuckDBPyConnection, users: list[int], k: int = 20):
    order = _popularity_order(_train_positives(con))
    seen = _seen(con, users)
    return {u: _top_k_by_rank(order, seen.get(u, set()), k) for u in users}


def recent_popularity(
    con: duckdb.DuckDBPyConnection, users: list[int], k: int = 20, days: int = RECENT_DAYS
):
    """Most-liked items in the last `days` days of train."""
    pos = _train_positives(con)
    pos = pos.filter(pl.col("ts") > pos["ts"].max() - days * 86400)
    order = _popularity_order(pos)
    seen = _seen(con, users)
    return {u: _top_k_by_rank(order, seen.get(u, set()), k) for u in users}


def item_knn(con: duckdb.DuckDBPyConnection, users: list[int], k: int = 20, batch: int = 256):
    """Item-item kNN: score(u, j) = sum over u's liked items i of cosine(i, j).

    Cosine is on the binary user x item matrix of train positives. It is applied as
    (U_b X^T) X with column-normalised X, which equals U_b S for the full item-item similarity S
    without materialising S (59k x 59k). Users with no liked train item fall back to popularity.
    """
    pos = _train_positives(con)
    pop_order = _popularity_order(pos)
    items, item_idx = np.unique(pos["movie_id"].to_numpy(), return_inverse=True)
    all_users, user_idx = np.unique(pos["user_id"].to_numpy(), return_inverse=True)
    x = sp.csr_matrix(
        (np.ones(len(pos), dtype=np.float32), (user_idx, item_idx)),
        shape=(len(all_users), len(items)),
    )
    norm = np.sqrt(np.asarray(x.sum(axis=0)).ravel())
    xn = (x @ sp.diags(1.0 / norm)).tocsr().astype(np.float32)
    xn_t = xn.T.tocsr()
    row_of = {u: i for i, u in enumerate(all_users)}
    item_col = {m: i for i, m in enumerate(items)}
    seen = _seen(con, users)

    recs: dict[int, np.ndarray] = {}
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        has_hist = [u for u in chunk if u in row_of]
        for u in chunk:
            if u not in row_of:
                recs[u] = _top_k_by_rank(pop_order, seen.get(u, set()), k)
        if not has_hist:
            continue
        hist = x[[row_of[u] for u in has_hist]]
        scores = ((hist @ xn_t) @ xn).toarray()
        for r, u in enumerate(has_hist):
            mask = [item_col[m] for m in seen.get(u, ()) if m in item_col]
            scores[r, mask] = -np.inf
            top = np.argpartition(-scores[r], k)[:k]
            recs[u] = items[top[np.argsort(-scores[r, top], kind="stable")]]
    return recs

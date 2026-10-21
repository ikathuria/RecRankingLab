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


class KnnScorer:
    """Item-item cosine kNN over the binary user x item matrix of train positives.

    score(u, j) = sum over u's liked items i of cosine(i, j), applied as (U_b X^T) X with
    column-normalised X. That equals U_b S for the full item-item similarity S without
    materialising S (59k x 59k).
    """

    def __init__(self, con: duckdb.DuckDBPyConnection):
        pos = _train_positives(con)
        self.pop_order = _popularity_order(pos)
        self.items, item_idx = np.unique(pos["movie_id"].to_numpy(), return_inverse=True)
        all_users, user_idx = np.unique(pos["user_id"].to_numpy(), return_inverse=True)
        self.x = sp.csr_matrix(
            (np.ones(len(pos), dtype=np.float32), (user_idx, item_idx)),
            shape=(len(all_users), len(self.items)),
        )
        norm = np.sqrt(np.asarray(self.x.sum(axis=0)).ravel())
        self.xn = (self.x @ sp.diags(1.0 / norm)).tocsr().astype(np.float32)
        self.xn_t = self.xn.T.tocsr()
        self.row_of = {u: i for i, u in enumerate(all_users)}
        self.item_col = {m: i for i, m in enumerate(self.items)}

    def batches(self, con: duckdb.DuckDBPyConnection, users: list[int], batch: int = 256):
        """Yield (users, scores) for users with a liked train item; scores[r, c] is for
        self.items[c], with the user's already-rated items set to -inf."""
        seen = _seen(con, users)
        with_hist = [u for u in users if u in self.row_of]
        for start in range(0, len(with_hist), batch):
            chunk = with_hist[start : start + batch]
            hist = self.x[[self.row_of[u] for u in chunk]]
            scores = ((hist @ self.xn_t) @ self.xn).toarray()
            for r, u in enumerate(chunk):
                scores[r, [self.item_col[m] for m in seen.get(u, ()) if m in self.item_col]] = -np.inf
            yield chunk, scores


def item_knn(con: duckdb.DuckDBPyConnection, users: list[int], k: int = 20, batch: int = 256):
    """Top-K by item-kNN score. Users with no liked train item fall back to popularity."""
    model = KnnScorer(con)
    recs: dict[int, np.ndarray] = {}
    for chunk, scores in model.batches(con, users, batch):
        for r, u in enumerate(chunk):
            top = np.argpartition(-scores[r], min(k, scores.shape[1] - 1))[:k]
            top = top[np.argsort(-scores[r, top], kind="stable")]
            recs[u] = model.items[top[np.isfinite(scores[r, top])]]
    missing = [u for u in users if u not in recs]
    seen = _seen(con, missing)
    for u in missing:
        recs[u] = _top_k_by_rank(model.pop_order, seen.get(u, set()), k)
    return recs

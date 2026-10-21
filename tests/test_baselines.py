import duckdb
import numpy as np
import pytest

from recsys.baselines import item_knn, popularity, recent_popularity
from recsys.split import make_split


@pytest.fixture
def con():
    """Users 1-4 like items 10,11 (u1 also 12); users 5,6 like 20,21. u1 has seen 10,11."""
    c = duckdb.connect()
    c.execute("CREATE TABLE ratings (user_id INT, movie_id INT, rating DOUBLE, ts BIGINT)")
    rows = []
    day = 86400
    # old era (day 0-9): popularity favours 20/21
    for u in range(1, 9):
        rows += [(u, m, 5.0, 1 * day) for m in ((10, 11) if u <= 4 else (20, 21))]
    rows += [(u, 12, 5.0, 2 * day) for u in (2, 3)]
    # recent era (day 100-109): item 30 is suddenly liked
    rows += [(u, 30, 5.0, (100 + u) * day) for u in (5, 6)]
    rows += [(1, 99, 5.0, 500 * day), (1, 98, 5.0, 501 * day)]  # later: only sets the cutoffs
    c.executemany("INSERT INTO ratings VALUES (?,?,?,?)", rows)
    make_split(c, train_q=0.9, val_q=0.95, min_history=1)
    return c


def test_popularity_excludes_seen_and_ranks_by_count(con):
    recs = popularity(con, [1, 5], k=3)
    assert 10 not in recs[1] and 11 not in recs[1]
    assert list(recs[1][:1]) == [20]  # 20 has 5 train positives, the most
    assert 20 not in recs[5] and 21 not in recs[5] and 30 not in recs[5]


def test_recent_popularity_prefers_recent_item(con):
    recs = recent_popularity(con, [1], k=2, days=20)  # window covers the last 20 days of train
    assert recs[1][0] == 30


def test_item_knn_recommends_cooccurring_unseen_item(con):
    recs = item_knn(con, [1], k=2)
    assert 10 not in recs[1] and 11 not in recs[1]
    assert recs[1][0] == 12  # co-liked with 10/11 by users 2 and 3
    assert len(recs[1]) == 2 and isinstance(recs[1], np.ndarray)


def test_item_knn_matches_dense_cosine_reference(con):
    users = [1, 2, 3, 4, 5, 6, 7, 8]
    recs = item_knn(con, users, k=3, batch=2)  # batch < len(users) exercises batching
    pos = con.execute("SELECT user_id, movie_id FROM split_ratings WHERE split='train'").fetchall()
    items = sorted({m for _, m in pos})
    x = np.zeros((9, max(items) + 1))
    for u, m in pos:
        x[u, m] = 1
    norm = np.sqrt(x.sum(axis=0))
    norm[norm == 0] = 1
    sim = (x / norm).T @ (x / norm)
    for u in users:
        score = x[u] @ sim
        score[x[u] > 0] = -np.inf
        score[[i for i in range(len(score)) if i not in items]] = -np.inf
        best = [i for i in np.argsort(-score, kind="stable")[:3] if score[i] > 0]
        assert list(recs[u][: len(best)]) == best

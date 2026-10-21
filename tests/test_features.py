import duckdb
import pytest

from recsys.features import (
    build_candidates,
    build_dataset,
    build_feature_tables,
    group_sizes,
    train_end,
)
from recsys.split import make_split

DAY = 86400


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("CREATE TABLE ratings (user_id INT, movie_id INT, rating DOUBLE, ts BIGINT)")
    c.execute("CREATE TABLE movies (movie_id BIGINT, title VARCHAR, genres VARCHAR)")
    c.executemany(
        "INSERT INTO movies VALUES (?,?,?)",
        [(m, f"Movie {m} ({1990 + m})", "Drama|Comedy" if m % 2 else "Drama") for m in range(1, 13)],
    )
    rows = []
    for u in (1, 2):  # train (days 1-8): items 1-8, liked only if m <= 4
        rows += [(u, m, 5.0 if m <= 4 else 3.0, m * DAY) for m in range(1, 9)]
    for u in (3, 4):  # train: items 5-12, all liked
        rows += [(u, m, 5.0, ((m - 1) % 8 + 1) * DAY) for m in range(5, 13)]
    rows += [(1, 9, 5.0, 20 * DAY), (2, 10, 4.0, 21 * DAY)]  # val positives
    rows += [(1, 11, 5.0, 30 * DAY), (2, 12, 5.0, 31 * DAY), (3, 12, 1.0, 32 * DAY)]  # test
    c.executemany("INSERT INTO ratings VALUES (?,?,?,?)", rows)
    make_split(c, train_q=0.86, val_q=0.9, min_history=3)
    return c


def test_features_use_only_ratings_up_to_as_of(con):
    as_of = 8 * DAY
    build_feature_tables(con, as_of)
    for table in ("user_features", "item_features"):
        assert con.execute(f"SELECT max(last_ts) FROM {table}").fetchone()[0] <= as_of
    before = con.execute("SELECT * FROM item_features ORDER BY movie_id").fetchall()
    ufeat = con.execute("SELECT * FROM user_features ORDER BY user_id").fetchall()
    genre = con.execute("SELECT * FROM user_genre ORDER BY ALL").fetchall()
    # extreme ratings after the cutoff must not change anything
    con.execute("INSERT INTO split_ratings VALUES (1, 1, 0.5, ?, 'test')", [9 * DAY])
    con.execute("INSERT INTO split_ratings VALUES (1, 12, 5.0, ?, 'test')", [40 * DAY])
    build_feature_tables(con, as_of)
    assert con.execute("SELECT * FROM item_features ORDER BY movie_id").fetchall() == before
    assert con.execute("SELECT * FROM user_features ORDER BY user_id").fetchall() == ufeat
    assert con.execute("SELECT * FROM user_genre ORDER BY ALL").fetchall() == genre


def test_feature_values(con):
    build_feature_tables(con, 8 * DAY)
    n, pos_rate, age = con.execute(
        "SELECT i_n_ratings, i_pos_rate, i_age_years FROM item_features WHERE movie_id = 2"
    ).fetchone()
    assert (n, pos_rate) == (2, 1.0)
    assert age == 1970 - 1992  # to_timestamp(8 days) is in 1970
    share = dict(con.execute("SELECT genre, share FROM user_genre WHERE user_id = 1").fetchall())
    assert share["Drama"] == pytest.approx(4 / 6)  # 4 liked items: Drama x4, Comedy x2
    assert share["Comedy"] == pytest.approx(2 / 6)


def test_dataset_groups_labels_and_exclusion(con):
    users = [1, 2, 3]
    df = build_dataset(con, users, "test", n=5)
    assert sum(group_sizes(df)) == df.height
    assert len(group_sizes(df)) == len(users)
    for u in users:  # a user's own train items are never candidates
        seen = {m for (m,) in con.execute(
            "SELECT movie_id FROM split_ratings WHERE split='train' AND user_id=?", [u]
        ).fetchall()}
        assert not (set(df.filter(df["user_id"] == u)["movie_id"]) & seen)
    assert df.filter((df["user_id"] == 1) & (df["movie_id"] == 11))["label"].to_list() == [1]
    assert df.filter((df["user_id"] == 2) & (df["movie_id"] == 12))["label"].to_list() == [1]
    assert df["label"].sum() == 2


def test_candidates_cover_unseen_items_only(con):
    cand = build_candidates(con, [1], n=3)
    assert len(cand) > 0 and set(cand["movie_id"]).isdisjoint(set(range(1, 9)))
    assert train_end(con) == 8 * DAY


def test_retrieved_candidates_replace_the_source_union(con):
    import numpy as np

    retrieved = {u: (np.array([11, 12]), np.array([0.9, 0.5])) for u in (1, 2)}
    df = build_dataset(con, [1, 2], "test", n=5, retrieved=retrieved)
    assert sorted(df["movie_id"].unique().to_list()) == [11, 12]
    assert df.height == 4
    row = df.filter((df["user_id"] == 1) & (df["movie_id"] == 12)).row(0, named=True)
    assert (row["tt_rank"], row["tt_score"], row["label"]) == (2, 0.5, 0)
    assert df.filter((df["user_id"] == 1) & (df["movie_id"] == 11))["label"][0] == 1  # u1 test +

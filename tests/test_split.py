import duckdb
import pytest

from recsys.split import eval_positives, make_split


@pytest.fixture
def con():
    """Synthetic ratings: u1 has long history, u2 is cold (2 train ratings), u3 is long."""
    c = duckdb.connect()
    c.execute("CREATE TABLE ratings (user_id INT, movie_id INT, rating DOUBLE, ts BIGINT)")
    rows = []
    for u in (1, 3):  # 30 ratings each, ts 0..29, then later ones
        rows += [(u, m, 4.0 if m % 2 == 0 else 2.0, m) for m in range(30)]
    rows += [(2, 100, 5.0, 1), (2, 101, 5.0, 2)]  # cold user, early
    rows += [(1, 200, 5.0, 40), (1, 201, 3.0, 41), (3, 200, 4.5, 90), (3, 202, 1.0, 95)]
    rows += [(2, 300, 5.0, 99)]  # cold user's late positive
    c.executemany("INSERT INTO ratings VALUES (?,?,?,?)", rows)
    return c


def test_test_timestamps_after_cutoff(con):
    info = make_split(con, 0.6, 0.8, min_history=20)
    mins = dict(con.execute("SELECT split, min(ts) FROM split_ratings GROUP BY split").fetchall())
    maxs = dict(con.execute("SELECT split, max(ts) FROM split_ratings GROUP BY split").fetchall())
    assert maxs["train"] <= info.train_end < mins["val"]
    assert maxs["val"] <= info.val_end < mins["test"]


def test_no_user_item_pair_in_two_splits(con):
    make_split(con, 0.6, 0.8, min_history=20)
    dupes = con.execute(
        "SELECT count(*) FROM (SELECT user_id, movie_id FROM split_ratings "
        "GROUP BY 1, 2 HAVING count(DISTINCT split) > 1)"
    ).fetchone()[0]
    assert dupes == 0


def test_cold_start_users_excluded_and_counted(con):
    info = make_split(con, 0.6, 0.8, min_history=20)
    users = set(con.execute("SELECT user_id FROM eligible_users").fetchnumpy()["user_id"])
    assert 2 not in users
    assert info.cold_start_users == 1
    for s in ("val", "test"):
        assert 2 not in set(eval_positives(con, s)["user_id"].to_list())


def test_positives_are_ratings_at_least_four(con):
    make_split(con, 0.6, 0.8, min_history=20)
    pos = set(
        zip(
            *eval_positives(con, "test")[["user_id", "movie_id"]].to_dict(as_series=False).values(),
            strict=True,
        )
    )
    expected = set(
        con.execute(
            "SELECT user_id, movie_id FROM split_ratings WHERE split='test' AND rating >= 4 "
            "AND user_id IN (SELECT user_id FROM eligible_users)"
        ).fetchall()
    )
    assert pos == expected
    assert (3, 202) not in pos  # rating 1.0

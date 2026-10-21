import duckdb
import pytest
import torch

from recsys.split import make_split
from recsys.twotower import prepare, retrieve, sampled_softmax_loss, train


@pytest.fixture
def con():
    """Users 1-6 like items 10-13 (u1 only rated 10,11); users 7-12 like items 20-23."""
    c = duckdb.connect()
    c.execute("CREATE TABLE ratings (user_id INT, movie_id INT, rating DOUBLE, ts BIGINT)")
    c.execute("CREATE TABLE movies (movie_id BIGINT, title VARCHAR, genres VARCHAR)")
    c.executemany(
        "INSERT INTO movies VALUES (?,?,?)",
        [(m, f"M{m} (2000)", "Drama" if m < 20 else "Comedy|Action") for m in (*range(10, 14), *range(20, 24))],
    )
    rows = []
    for u in range(1, 13):
        items = (10, 11, 12, 13) if u <= 6 else (20, 21, 22, 23)
        rows += [(u, m, 5.0, 86400) for m in (items[:2] if u == 1 else items)]
    rows += [(1, 99, 5.0, 500 * 86400), (1, 98, 5.0, 501 * 86400)]  # only set the cutoffs
    c.execute("INSERT INTO movies VALUES (99, 'M99 (2000)', 'Drama'), (98, 'M98 (2000)', 'Drama')")
    c.executemany("INSERT INTO ratings VALUES (?,?,?,?)", rows)
    make_split(c, train_q=0.9, val_q=0.95, min_history=1)
    return c


def test_prepare_maps_train_positives_only(con):
    data = prepare(con)
    assert data.n_users == 12 and data.n_items == 8  # items 98/99 are not train items
    assert data.pairs.shape == (46, 2)
    assert data.genres.shape == (8, 3)  # Drama, Comedy, Action
    assert data.genres.sum(axis=1).tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_loss_decreases_and_embeddings_have_expected_shape(con):
    data = prepare(con)
    model, hist = train(data, dim=16, hidden=32, epochs=30, batch=16, lr=1e-2)
    assert hist[-1]["loss"] < hist[0]["loss"] * 0.8
    assert model.user_vec(torch.arange(data.n_users)).shape == (12, 16)
    v = model.item_vec(torch.arange(data.n_items))
    assert v.shape == (8, 16)
    assert torch.allclose(v.norm(dim=1), torch.ones(8), atol=1e-5)  # unit length


def test_retrieve_excludes_seen_and_finds_own_cluster(con):
    data = prepare(con)
    model, _ = train(data, dim=16, hidden=32, epochs=60, batch=16, lr=1e-2)
    out = retrieve(model, data, con, [1, 7], k=3)
    items, scores = out[1]
    assert not {10, 11} & set(items)  # u1 rated these in train
    assert set(items[:2]) == {12, 13}  # same cluster as u1's taste
    assert list(scores) == sorted(scores, reverse=True)
    assert 20 not in out[7][0] and set(out[7][0]) <= {10, 11, 12, 13, 20, 21, 22, 23}


def test_duplicate_items_in_batch_are_not_negatives():
    u = torch.eye(2).repeat(2, 1)  # 4 users, users 0 and 2 identical
    v = u.clone()
    items = torch.tensor([5, 6, 5, 6])  # rows 0 and 2 share an item
    log_q = torch.zeros(10)
    loss = sampled_softmax_loss(u, v, items, log_q, temperature=0.1)
    assert torch.isfinite(loss) and loss < 1e-3  # without the mask, the twin caps loss at log 2

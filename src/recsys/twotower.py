"""Two-tower retrieval: user and item towers trained with an in-batch sampled softmax.

The user tower embeds the user id; the item tower embeds the item id plus its genres. Both are
L2-normalised, so the score is a cosine similarity divided by a temperature. Each batch of
(user, liked item) pairs uses the other items in the batch as negatives, with the logQ correction
for the popularity-skewed sampling of those negatives (Yi et al., RecSys 2019). Only train data
is read. Retrieval is an exact dot product over all items (docs/01-evaluation-protocol.md).
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.baselines import _popularity_order, _seen, _top_k_by_rank, _train_positives

TEMPERATURE = 0.05


@dataclass
class Data:
    """Index maps and training pairs. `pairs` rows are (user index, item index)."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    pairs: np.ndarray
    genres: np.ndarray  # (n_items, n_genres) multi-hot, float32
    pop_order: np.ndarray  # item ids by train popularity, for users with no liked train item

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def prepare(con: duckdb.DuckDBPyConnection) -> Data:
    """Train positives as index pairs; the item vocabulary is the retrieval candidate pool."""
    pos = _train_positives(con)
    user_ids, u_idx = np.unique(pos["user_id"].to_numpy(), return_inverse=True)
    item_ids, i_idx = np.unique(pos["movie_id"].to_numpy(), return_inverse=True)
    movies = dict(con.execute("SELECT movie_id, genres FROM movies").fetchall())
    names = sorted({g for m in item_ids for g in movies.get(int(m), "").split("|") if g})
    col = {g: j for j, g in enumerate(names)}
    genres = np.zeros((len(item_ids), len(names)), dtype=np.float32)
    for r, m in enumerate(item_ids):
        for g in movies.get(int(m), "").split("|"):
            if g:
                genres[r, col[g]] = 1.0
    return Data(
        user_ids, item_ids, np.stack([u_idx, i_idx], axis=1), genres, _popularity_order(pos)
    )


def _mlp(dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, genres: np.ndarray, dim: int = 64,
                 hidden: int = 128):  # fmt: skip
        super().__init__()
        self.dim = dim
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.genre_proj = nn.Linear(genres.shape[1], dim, bias=False)
        self.register_buffer("genres", torch.as_tensor(genres))
        self.user_mlp = _mlp(dim, hidden)
        self.item_mlp = _mlp(dim, hidden)
        for emb in (self.user_emb, self.item_emb):
            nn.init.normal_(emb.weight, std=0.1)

    def user_vec(self, idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_mlp(self.user_emb(idx)), dim=-1)

    def item_vec(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(idx) + self.genre_proj(self.genres[idx])
        return F.normalize(self.item_mlp(x), dim=-1)


def sampled_softmax_loss(
    u: torch.Tensor, v: torch.Tensor, items: torch.Tensor, log_q: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:  # fmt: skip
    """Cross-entropy of each user's own item against the other items in the batch.

    logits[i, j] = u_i . v_j / T - log q(item_j), where q is the probability that an item is the
    positive of a random training pair. The same item at two batch positions is not a negative.
    """
    logits = u @ v.T / temperature - log_q[items][None, :]
    dup = (items[:, None] == items[None, :]) & ~torch.eye(len(items), dtype=torch.bool)
    logits = logits.masked_fill(dup, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(items)))


def train(
    data: Data, *, dim: int = 64, hidden: int = 128, epochs: int = 5, batch: int = 4096,
    lr: float = 2e-3, seed: int = 0, on_epoch=None,
) -> tuple[TwoTower, list[dict]]:  # fmt: skip
    """Train and return (model, history). `on_epoch(model, epoch) -> dict` may add metrics (e.g.
    validation recall) to that epoch's history row; if it returns "select", the epoch with the
    highest value of that metric is restored at the end."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoTower(data.n_users, data.n_items, data.genres, dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    counts = np.bincount(data.pairs[:, 1], minlength=data.n_items).clip(min=1)
    log_q = torch.as_tensor(np.log(counts / counts.sum()), dtype=torch.float32)
    pairs = torch.as_tensor(data.pairs)
    history: list[dict] = []
    best: tuple[float, dict] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.as_tensor(rng.permutation(len(pairs)))
        losses = []
        for start in range(0, len(order) - batch + 1, batch):  # drop the ragged last batch
            b = pairs[order[start : start + batch]]
            loss = sampled_softmax_loss(model.user_vec(b[:, 0]), model.item_vec(b[:, 1]), b[:, 1], log_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        model.eval()
        if on_epoch:
            row |= on_epoch(model, epoch)
        history.append(row)
        print(row, flush=True)
        sel = row.get("select")
        if sel is not None and (best is None or sel > best[0]):
            best = (sel, {k: v.clone() for k, v in model.state_dict().items()})
    if best:
        model.load_state_dict(best[1])
    model.eval()
    return model, history


@torch.no_grad()
def retrieve(
    model: TwoTower, data: Data, con: duckdb.DuckDBPyConnection, users: list[int], k: int = 200,
    batch: int = 512,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """Exact top-k by dot product, excluding items the user rated in train.

    Returns {user_id: (item ids best first, scores)}. Users with no liked train item have no
    embedding and fall back to popularity (score 0), like the kNN baseline.
    """
    model.eval()
    items = model.item_vec(torch.arange(data.n_items))
    row_of = {int(u): i for i, u in enumerate(data.user_ids)}
    col_of = {int(m): i for i, m in enumerate(data.item_ids)}
    seen = _seen(con, users)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with_emb = [u for u in users if u in row_of]
    for start in range(0, len(with_emb), batch):
        chunk = with_emb[start : start + batch]
        scores = model.user_vec(torch.as_tensor([row_of[u] for u in chunk])) @ items.T
        for r, u in enumerate(chunk):
            cols = [col_of[m] for m in seen.get(u, ()) if m in col_of]
            scores[r, cols] = float("-inf")
        top = torch.topk(scores, min(k, data.n_items), dim=1)
        for r, u in enumerate(chunk):
            ok = torch.isfinite(top.values[r])
            out[u] = (data.item_ids[top.indices[r][ok].numpy()], top.values[r][ok].numpy())
    for u in users:
        if u not in out:
            fallback = _top_k_by_rank(data.pop_order, seen.get(u, set()), k)
            out[u] = (fallback, np.zeros(len(fallback), dtype=np.float32))
    return out

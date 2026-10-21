"""Top-K ranking metrics (per user) and bootstrap CIs over users.

Binary relevance. Recall divides by all of a user's positives; NDCG and MAP use the ideal list
truncated at K, so a perfect ranking scores 1.0.
"""

from collections.abc import Mapping, Sequence

import numpy as np


def _hits(recs: Sequence[int], relevant: set[int], k: int) -> np.ndarray:
    return np.fromiter((i in relevant for i in list(recs)[:k]), dtype=float)


def recall_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    return float(_hits(recs, relevant, k).sum() / len(relevant))


def ndcg_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    hits = _hits(recs, relevant, k)
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    ideal = (1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum()
    return float(dcg / ideal)


def ap_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    hits = _hits(recs, relevant, k)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / min(len(relevant), k))


def coverage(recs: Mapping[int, Sequence[int]], n_items: int, k: int) -> float:
    """Share of the catalogue (`n_items`) that appears in at least one user's top-K."""
    return len({i for r in recs.values() for i in list(r)[:k]}) / n_items


def per_user_metrics(
    recs: Mapping[int, Sequence[int]], truth: Mapping[int, set[int]], ks: Sequence[int] = (10, 20)
) -> dict[str, np.ndarray]:
    """Per-user metric arrays over the users in `truth` (those with >= 1 positive).

    A user missing from `recs` gets an empty list (scores 0), so no one is silently dropped.
    """
    out: dict[str, list[float]] = {f"{m}@{k}": [] for k in ks for m in ("recall", "ndcg", "map")}
    for user, relevant in truth.items():
        r = recs.get(user, [])
        for k in ks:
            out[f"recall@{k}"].append(recall_at_k(r, relevant, k))
            out[f"ndcg@{k}"].append(ndcg_at_k(r, relevant, k))
            out[f"map@{k}"].append(ap_at_k(r, relevant, k))
    return {name: np.asarray(v) for name, v in out.items()}


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lower, upper): percentile bootstrap CI of the mean, resampling users."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """CI of mean(a - b) with the same users resampled for both models (paired)."""
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n_boot, alpha, seed)

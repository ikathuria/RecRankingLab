import math

import numpy as np
import pytest

from recsys.metrics import (
    ap_at_k,
    bootstrap_ci,
    coverage,
    ndcg_at_k,
    paired_bootstrap_diff,
    per_user_metrics,
    recall_at_k,
)

RECS = [1, 2, 3, 4]
REL = {2, 4, 9}  # hits at ranks 2 and 4; one positive never recommended


def test_recall():
    assert recall_at_k(RECS, REL, 4) == pytest.approx(2 / 3)
    assert recall_at_k(RECS, REL, 2) == pytest.approx(1 / 3)  # only rank 2 counts


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RECS, REL, 4) == pytest.approx(dcg / idcg)


def test_ap_hand_computed():
    # precision at hits: 1/2 and 2/4; divided by min(|rel|, K) = 3
    assert ap_at_k(RECS, REL, 4) == pytest.approx((0.5 + 0.5) / 3)


def test_perfect_ranking_scores_one_even_with_more_positives_than_k():
    rel = set(range(10))
    assert ndcg_at_k(list(range(10)), rel, 3) == pytest.approx(1.0)
    assert ap_at_k(list(range(10)), rel, 3) == pytest.approx(1.0)
    assert recall_at_k(list(range(10)), rel, 3) == pytest.approx(0.3)


def test_no_hits_is_zero():
    assert recall_at_k([7, 8], {1}, 2) == ndcg_at_k([7, 8], {1}, 2) == ap_at_k([7, 8], {1}, 2) == 0


def test_coverage():
    assert coverage({1: [1, 2], 2: [2, 3]}, n_items=10, k=2) == 0.3
    assert coverage({1: [1, 2], 2: [2, 3]}, n_items=10, k=1) == 0.2


def test_per_user_missing_user_scores_zero():
    m = per_user_metrics({1: [5]}, {1: {5}, 2: {6}}, ks=(1,))
    assert list(m["recall@1"]) == [1.0, 0.0]


def test_bootstrap_ci_constant_and_coverage():
    assert bootstrap_ci(np.full(50, 0.3)) == pytest.approx((0.3, 0.3, 0.3))
    vals = np.random.default_rng(1).binomial(1, 0.2, size=2000).astype(float)
    mean, lo, hi = bootstrap_ci(vals, seed=3)
    assert lo < mean < hi
    assert hi - lo == pytest.approx(2 * 1.96 * vals.std() / math.sqrt(2000), rel=0.15)


def test_bootstrap_is_deterministic_and_paired_diff_uses_same_users():
    a = np.array([1.0, 0.0, 1.0, 1.0])
    assert bootstrap_ci(a, seed=5) == bootstrap_ci(a, seed=5)
    mean, lo, hi = paired_bootstrap_diff(a, a)
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)

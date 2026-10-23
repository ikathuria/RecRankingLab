import numpy as np
import polars as pl

from recsys.abtest import (
    cuped_adjust,
    diff_test,
    engagement_prob,
    mde_for_n,
    power_for_n,
    run_study,
    sample_size_per_arm,
    simulate_experiment,
    srm_check,
)


def synthetic(n=4000, effect=0.5, rho=0.7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    treat = rng.permutation(np.arange(n) % 2)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n) + effect * treat
    return y, x, treat


def test_power_functions_agree():
    n = sample_size_per_arm(0.05, 0.5)
    assert abs(mde_for_n(n, 0.5) - 0.05) < 1e-3
    assert abs(power_for_n(0.05, n, 0.5) - 0.8) < 0.01
    assert sample_size_per_arm(0.05, 0.5, variance_ratio=0.5) < n


def test_recovers_injected_effect():
    y, _, t = synthetic(n=20000, effect=0.5)
    r = diff_test(y, t)
    assert r.ci[0] < 0.5 < r.ci[1] and r.p < 1e-6


def test_cuped_reduces_variance_and_stays_unbiased():
    y, x, t = synthetic(rho=0.7)
    adj, theta = cuped_adjust(y, x)
    assert adj.var() <= y.var()
    assert abs(theta - 0.7) < 0.05
    naive, cuped = diff_test(y, t), diff_test(adj, t)
    assert cuped.se < naive.se
    assert abs(cuped.diff - 0.5) < 4 * cuped.se


def test_cuped_with_uncorrelated_covariate_does_not_hurt():
    y, _, t = synthetic(rho=0.0)
    x = np.random.default_rng(5).normal(size=len(y))
    adj, _ = cuped_adjust(y, x)
    assert diff_test(adj, t).se <= diff_test(y, t).se * 1.01


def test_srm_flags_skewed_split_only():
    assert not srm_check(5000, 5010)[1]
    p, bad = srm_check(5000, 5400)
    assert bad and p < 0.001


def test_engagement_prob_monotone_in_hits():
    q = engagement_prob(np.arange(0, 11, dtype=float))
    assert q[0] == 0.02 * 0 + 1 - 0.98**10 and (np.diff(q) > 0).all() and q[-1] < 1


def population(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    q_c = np.clip(0.3 + 0.08 * x, 0.01, 0.99)
    return pl.DataFrame({"pre_activity": x, "q_control": q_c, "q_treatment": np.clip(q_c + 0.05, 0, 1)})


def test_aa_has_no_effect_and_ab_recovers_it():
    pop = population()
    rng = np.random.default_rng(0)
    aa = [diff_test(d["y"], d["treat"]).diff for d in
          (simulate_experiment(rng, pop, 6000, aa=True) for _ in range(200))]
    ab = [diff_test(d["y"], d["treat"]).diff for d in
          (simulate_experiment(rng, pop, 6000, aa=False) for _ in range(200))]
    assert abs(np.mean(aa)) < 0.005
    assert abs(np.mean(ab) - 0.05) < 0.005


def test_study_false_positive_rate_near_alpha_and_cuped_helps():
    res = run_study(population(), n_users=4000, n_sims=300, seed=2)
    assert 0.02 < res["aa"]["reject_naive"] < 0.09
    assert 0.02 < res["aa"]["reject_cuped"] < 0.09
    assert res["ab"]["mean_se_cuped"] <= res["ab"]["mean_se_naive"]
    assert res["ab"]["reject_cuped"] >= res["ab"]["reject_naive"] - 0.05

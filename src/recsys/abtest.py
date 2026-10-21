"""Milestone 6: a SIMULATED A/B test: power analysis, CUPED, SRM check, A/A and A/B studies.

Nothing here is a real experiment. Per-user engagement probabilities are derived from held-out
ratings and the models' top-10 lists under the assumptions in docs/03-ab-simulation.md, then users
are re-sampled and randomised 50/50 in software. Usage:
python -m recsys.abtest [inputs|study|all] [--limit N] [--sims N]
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
INPUTS = RESULTS_DIR / "ab_inputs.parquet"
STUDY = RESULTS_DIR / "ab_study.json"
PLOT = Path(__file__).resolve().parents[2] / "docs" / "03-ab-power.png"
DOC = Path(__file__).resolve().parents[2] / "docs" / "03-ab-simulation.md"
MARKER = "## Simulation results"
TOP_N = 10
P_CLICK = 0.30  # chance a shown item the user rated >= 4 later is clicked (assumption)
P_NOISE = 0.02  # chance a shown item outside that set is clicked (assumption)
SRM_ALPHA = 0.001


# --- statistics ------------------------------------------------------------------------------


def sample_size_per_arm(mde: float, sd: float, alpha: float = 0.05, power: float = 0.8,
                        variance_ratio: float = 1.0) -> int:
    """Users per arm to detect an absolute difference `mde` in means (two-sided z-test).

    `variance_ratio` scales the outcome variance, e.g. 1 - rho^2 for CUPED."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return math.ceil(2 * (z * sd) ** 2 * variance_ratio / mde**2)


def mde_for_n(n_per_arm: int, sd: float, alpha: float = 0.05, power: float = 0.8,
              variance_ratio: float = 1.0) -> float:
    """Smallest absolute difference detectable with `n_per_arm` users per arm."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * sd * math.sqrt(2 * variance_ratio / n_per_arm)


def power_for_n(delta: float, n_per_arm: int, sd: float, alpha: float = 0.05,
                variance_ratio: float = 1.0) -> float:
    se = sd * math.sqrt(2 * variance_ratio / n_per_arm)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - abs(delta) / se) + stats.norm.cdf(-z - abs(delta) / se))


@dataclass
class TestResult:
    diff: float
    se: float
    ci: tuple[float, float]
    p: float


def diff_test(y: np.ndarray, treat: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Treatment minus control difference in means with a Welch z interval."""
    a, b = y[treat == 1], y[treat == 0]
    diff = float(a.mean() - b.mean())
    se = float(math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    z = stats.norm.ppf(1 - alpha / 2)
    return TestResult(diff, se, (diff - z * se, diff + z * se), float(2 * stats.norm.sf(abs(diff / se))))


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: y - theta (x - mean(x)), theta = cov(x, y) / var(x) pooled over both arms.

    x must be measured before randomisation, so it cannot depend on the arm."""
    theta = float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
    return y - theta * (x - x.mean()), theta


def srm_check(n_control: int, n_treatment: int, expected: float = 0.5) -> tuple[float, bool]:
    """Chi-square test of the arm counts against the intended split. Returns (p, mismatch)."""
    n = n_control + n_treatment
    p = float(stats.chisquare([n_control, n_treatment], [n * (1 - expected), n * expected]).pvalue)
    return p, p < SRM_ALPHA


# --- simulation ------------------------------------------------------------------------------


def engagement_prob(hits: np.ndarray, n_shown: int = TOP_N) -> np.ndarray:
    """P(user clicks >= 1 shown item) when `hits` of the `n_shown` items are later-liked."""
    return 1 - (1 - P_CLICK) ** hits * (1 - P_NOISE) ** (n_shown - hits)


def simulate_experiment(rng: np.random.Generator, pop: pl.DataFrame, n_users: int, *, aa: bool,
                        extra_effect: float = 0.0) -> dict[str, np.ndarray]:
    """Draw `n_users` from the population (with replacement), randomise 50/50, draw engagement.

    A/A: both arms see the control probabilities. `extra_effect` is added to the treatment
    probability (clipped to [0, 1]) to inject a known effect on top of the modelled one."""
    idx = rng.integers(0, pop.height, n_users)
    x = pop["pre_activity"].to_numpy()[idx]
    q_c = pop["q_control"].to_numpy()[idx]
    q_t = q_c if aa else pop["q_treatment"].to_numpy()[idx]
    q_t = np.clip(q_t + extra_effect, 0, 1)
    treat = rng.permutation(np.arange(n_users) % 2)  # exactly balanced; SRM tested separately
    y = (rng.random(n_users) < np.where(treat == 1, q_t, q_c)).astype(float)
    return {"y": y, "x": x, "treat": treat}


def run_study(pop: pl.DataFrame, n_users: int, n_sims: int, alpha: float = 0.05,
              seed: int = 0) -> dict:
    """N A/A tests (false-positive rate) and N A/B tests (empirical power), naive and CUPED."""
    rng = np.random.default_rng(seed)
    out: dict = {"n_users": n_users, "n_sims": n_sims, "alpha": alpha}
    for kind, aa in (("aa", True), ("ab", False)):
        rej = {"naive": 0, "cuped": 0}
        se = {"naive": [], "cuped": []}
        est = {"naive": [], "cuped": []}
        srm = 0
        for _ in range(n_sims):
            d = simulate_experiment(rng, pop, n_users, aa=aa)
            adj, _ = cuped_adjust(d["y"], d["x"])
            for name, y in (("naive", d["y"]), ("cuped", adj)):
                r = diff_test(y, d["treat"], alpha)
                rej[name] += r.p < alpha
                se[name].append(r.se)
                est[name].append(r.diff)
            srm += srm_check(int((d["treat"] == 0).sum()), int((d["treat"] == 1).sum()))[1]
        out[kind] = {
            "reject_naive": rej["naive"] / n_sims, "reject_cuped": rej["cuped"] / n_sims,
            "mean_se_naive": float(np.mean(se["naive"])), "mean_se_cuped": float(np.mean(se["cuped"])),
            "mean_est_naive": float(np.mean(est["naive"])), "mean_est_cuped": float(np.mean(est["cuped"])),
            "srm_flags": srm,
        }
    return out


def plan(pop: pl.DataFrame, alpha: float = 0.05, power: float = 0.8) -> dict:
    """Power analysis from the population: effect, outcome sd, correlation with the covariate."""
    q_c, q_t = pop["q_control"].to_numpy(), pop["q_treatment"].to_numpy()
    delta = float((q_t - q_c).mean())
    # y ~ Bernoulli(q) with q drawn from the population, so marginally Bernoulli(mean q)
    var_c, var_t = q_c.mean() * (1 - q_c.mean()), q_t.mean() * (1 - q_t.mean())
    sd = float(np.sqrt((var_c + var_t) / 2))
    x = pop["pre_activity"].to_numpy()
    rho = float(np.cov(x, q_c)[0, 1] / (x.std(ddof=1) * math.sqrt(var_c)))
    ratio = 1 - rho**2
    return {"delta": delta, "sd": sd, "rho": rho, "variance_ratio": ratio,
            "n_per_arm": sample_size_per_arm(delta, sd, alpha, power),
            "n_per_arm_cuped": sample_size_per_arm(delta, sd, alpha, power, ratio)}


# --- inputs from the models ------------------------------------------------------------------


def build_inputs(limit: int | None) -> pl.DataFrame:
    """Per-user hits in the control (popularity) and treatment (LightGBM ranker) top-10."""
    import duckdb

    from recsys import ranker
    from recsys.data import DB_PATH
    from recsys.features import FEATURES

    con = duckdb.connect(str(DB_PATH), read_only=True)
    val_users, _ = ranker.eval_users(con, "val", limit)
    test_users, _ = ranker.eval_users(con, "test", limit)
    val, test = ranker.dataset(con, "val", val_users), ranker.dataset(con, "test", test_users)
    rng = np.random.default_rng(0)  # same early-stopping split as ranker.main
    es_users = set(rng.choice(val_users, size=max(1, len(val_users) // 5), replace=False).tolist())
    in_es = pl.col("user_id").is_in(list(es_users))
    model = ranker.fit(ranker.drop_empty_groups(val.filter(~in_es)),
                       ranker.drop_empty_groups(val.filter(in_es)), FEATURES)
    scored = test.with_columns(pl.Series("score", model.predict(test.select(FEATURES).to_numpy())))
    treat = (scored.sort(["user_id", "score"], descending=[False, True]).group_by("user_id", maintain_order=True)
             .head(TOP_N).group_by("user_id").agg(pl.col("label").sum().alias("hits_treatment")))
    ctrl = (test.filter(pl.col("pop_rank") <= TOP_N).group_by("user_id")
            .agg(pl.col("label").sum().alias("hits_control")))
    pre = test.group_by("user_id").agg(pl.col("u_n_ratings").first())
    df = pre.join(ctrl, on="user_id", how="left").join(treat, on="user_id", how="left").fill_null(0)
    return df.with_columns(
        pre_activity=pl.col("u_n_ratings").log1p(),
        q_control=pl.Series(engagement_prob(df["hits_control"].to_numpy().astype(float))),
        q_treatment=pl.Series(engagement_prob(df["hits_treatment"].to_numpy().astype(float))),
    ).sort("user_id")


# --- report ----------------------------------------------------------------------------------


def make_plot(pl_: dict, study: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = np.linspace(500, 40_000, 80).astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, ratio in (("unadjusted", 1.0), (f"CUPED (rho={pl_['rho']:.2f})", pl_["variance_ratio"])):
        ax.plot(ns, [mde_for_n(int(n), pl_["sd"], variance_ratio=ratio) for n in ns], label=label)
    ax.axhline(pl_["delta"], color="grey", ls="--", label=f"modelled effect {pl_['delta']:.3f}")
    ax.set(xlabel="users per arm", ylabel="minimum detectable effect (absolute, 80% power, α=0.05)",
           title="SIMULATED A/B test: MDE vs sample size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT, dpi=130)


def write_doc(pl_: dict, study: dict) -> None:
    aa, ab, n = study["aa"], study["ab"], study["n_sims"]
    planned = power_for_n(pl_["delta"], study["n_users"] // 2, pl_["sd"])
    planned_c = power_for_n(pl_["delta"], study["n_users"] // 2, pl_["sd"], variance_ratio=pl_["variance_ratio"])
    var_cut = 1 - (ab["mean_se_cuped"] / ab["mean_se_naive"]) ** 2
    section = f"""{MARKER}

**All numbers below are SIMULATED. No real users were randomised.**

Population: {study['pop_users']:,} held-out test users. Modelled effect (treatment − control mean
engagement probability): {pl_['delta']:.4f}. Outcome sd {pl_['sd']:.3f}; correlation of the covariate
(log prior ratings) with the outcome ρ = {pl_['rho']:.3f}, so CUPED should cut variance by about
{pl_['rho']**2:.1%}.

Power analysis (α = 0.05, power 0.8): {pl_['n_per_arm']:,} users per arm unadjusted,
{pl_['n_per_arm_cuped']:,} with CUPED.

Study: {n:,} A/A and {n:,} A/B experiments of {study['n_users']:,} users each (seed {study['seed']}).

| | A/A rejection rate (target {study['alpha']}) | A/B rejection rate (empirical power) | planned power | mean SE |
|---|---|---|---|---|
| unadjusted | {aa['reject_naive']:.3f} | {ab['reject_naive']:.3f} | {planned:.3f} | {ab['mean_se_naive']:.5f} |
| CUPED | {aa['reject_cuped']:.3f} | {ab['reject_cuped']:.3f} | {planned_c:.3f} | {ab['mean_se_cuped']:.5f} |

Observed variance reduction from CUPED: {var_cut:.1%}. SRM flags (p < {SRM_ALPHA}) across A/A runs:
{aa['srm_flags']} of {n:,}. Assignment is exactly balanced in the simulation, so the check is only
exercised by unit tests that inject a skewed split.

![MDE vs sample size](03-ab-power.png)
"""
    text = DOC.read_text() if DOC.exists() else ""
    head = text.split(MARKER)[0].rstrip() + "\n\n" if text else ""
    DOC.write_text(head + section)


def main() -> None:
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["inputs", "study", "all"])
    ap.add_argument("--limit", type=int, help="smoke test on the first N users per split")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--users", type=int, help="users per experiment (default: planned n, both arms)")
    a = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    path = INPUTS if not a.limit else RESULTS_DIR / f"ab_inputs_{a.limit}.parquet"
    if a.stage in ("inputs", "all"):
        build_inputs(a.limit).write_parquet(path)
        print("wrote", path, flush=True)
    if a.stage in ("study", "all"):
        pop = pl.read_parquet(path)
        pl_ = plan(pop)
        n_users = a.users or 2 * pl_["n_per_arm"]
        study = run_study(pop, n_users, a.sims) | {"pop_users": pop.height, "seed": 0}
        print(json.dumps({"plan": pl_, "study": study}, indent=1))
        if not a.limit:
            STUDY.write_text(json.dumps({"plan": pl_, "study": study}, indent=1))
            make_plot(pl_, study)
            write_doc(pl_, study)


if __name__ == "__main__":
    main()

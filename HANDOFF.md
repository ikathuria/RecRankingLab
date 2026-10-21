# Handoff: 2026-10-21 · M1-M6 done, M7 next

**Goal:** Portfolio data-science project: two-stage MovieLens 25M recommender with leakage-safe eval and a simulated A/B test. See PLAN.md (8 milestones).
**Status:** M1-M5 merged to `main`. M6 on branch `claude/auto-2026-10-21-m6-abtest` (PR open, not merged).

## Next step
Merge the M6 PR once CI is green, then **M7** (Streamlit app + README; see PLAN.md). The app should read `results/ab_inputs.parquet`, `results/ab_study.json` and `docs/03-ab-power.png`.

## Done (M6)
- `src/recsys/abtest.py` (`python -m recsys.abtest [inputs|study|all]`; `inputs` retrains the M4 ranker, a few min), `tests/test_abtest.py` (8 tests; 36 total pass, lint clean), `docs/03-ab-simulation.md`.
- Simulated: control popularity top-10 vs LightGBM top-10, 3,833 test users resampled, engagement = P(click >= 1 shown item) with assumed click probs 0.30 (later-liked) / 0.02 (other). Modelled effect 0.059; 946 users/arm for 80% power.
- 1,000 A/A: false-positive rate 0.042 (target 0.05). 1,000 A/B at n=1,892: power 0.787 vs planned 0.800. 0 SRM flags.

## Decisions
- Population is resampled with replacement from the real test users; the effect size depends on the assumed click probabilities, not just the data.
- Pre-period covariate = log1p(`u_n_ratings`).

## Open threads
- **CUPED gives ~0.1% variance reduction** (rho = -0.03): the covariate does not predict engagement here. The "CUPED variance <= unadjusted" check is demonstrated on synthetic data in tests, not on this population. A better covariate (pre-period positive rate) is untried.
- SRM is only exercised by unit tests (simulation assigns exactly 50/50).
- M5 threads still open: more epochs, union of candidates, `i_age_years` ablation.

## Gotchas
- **torch and lightgbm in one process segfault on macOS** (libomp clash). `abtest` imports lightgbm only, never torch.
- `docs/03-ab-simulation.md`: the script rewrites everything after `## Simulation results`; edit the hand-written part above it only (or the CUPED note will be lost on rerun).
- Earlier: LightGBM needs `brew install libomp`; MLflow metric names can't contain `@`; caches in `data/cache`; moving the repo breaks `.venv`; `timeout`/`uv` not installed.

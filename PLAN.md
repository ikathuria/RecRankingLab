# RecRankingLab

> A two-stage movie recommender (candidate generation + learned ranker) with an honest offline evaluation and a simulated A/B test (power analysis, CUPED), built on MovieLens 25M for a data-science portfolio.

---

## Viability Summary

| | |
|---|---|
| **Build it?** | yes. Portfolio project; closes the SQL / statistics / A/B testing / recsys / MLOps gap in the JobApply profile (PLAN M43) |
| **Market** | n/a. MovieLens recsys demos are common, so the value is rigor, not novelty |
| **Demand** | n/a. Recruiter-keyword and interview-talking-point value, not user demand |
| **Direction** | neutral |
| **Feasibility** | easy to medium. The risky part is correctness (leakage, valid A/B simulation), not tooling |
| **Free to build** | yes. Public dataset, open-source stack, laptop-sized |
| **Monetization** | portfolio project |

**In two sentences:** Differentiate on evaluation rigor (time-based split, leakage-free features, significance tests, a clearly labelled simulation), not on model novelty. Skipped the idea-research skill: there is no market question for a portfolio project.

---

## Research Findings

- **Wedge:** most MovieLens repos stop at RMSE or a single NDCG number. This one reports Recall@K / NDCG@K against popularity and item-kNN baselines with confidence intervals, then runs a power-analysed, CUPED-adjusted simulated A/B test.
- **Dataset:** MovieLens 25M (GroupLens, https://files.grouplens.org/datasets/movielens/ml-25m.zip, ~250 MB, 25M ratings with timestamps). Non-commercial research use: do not redistribute the raw data; the repo downloads it.
- **Spike:** none needed. Leakage is the main correctness risk; see Risks.
- **Cost flags:** none.
- **Open question:** how much does the ranker add over a strong popularity + item-kNN baseline? Report the answer even if it is small.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Feature leakage (features computed with data after the prediction cutoff) | High | High | Time-based split; every feature SQL takes an `as_of` timestamp; a test asserts no feature row uses data later than its cutoff |
| Ranker barely beats popularity | Medium | Medium | Report honestly; ablate features to show what helps |
| Simulated A/B is mistaken for a real experiment | Medium | High | Label "simulated" in README, dashboard and plots; state the generative assumptions |
| Training cost on a laptop | Low | Low | Sample users for dev (`--sample 0.05`); full data only for final run |
| MovieLens licence | Low | Medium | Download script only; raw data gitignored; cite GroupLens |

---

## Tech Stack

Python 3.12 (3.14 may lack wheels for the ML libs). `.venv` + `pip` + Makefile, like JobApply (uv is not installed on this machine).

| Layer | Choice | Version | Why |
|---|---|---|---|
| SQL / features | DuckDB | 1.5.6 | Fast analytical SQL on a laptop, reads CSV/Parquet directly |
| Dataframes | Polars | 2.0.0 | Fast, pairs with DuckDB via Arrow |
| Ranker | LightGBM (LambdaRank) | 4.7.0 | Standard learning-to-rank baseline |
| Two-tower retrieval | PyTorch | 2.14.1 | Candidate generation; CPU is enough at this scale |
| Tracking | MLflow | 3.17.2 | Runs, params, metrics, model registry (local file store) |
| Stats | SciPy / statsmodels | 1.18.1 / 0.15.0 | Power analysis, tests, CUPED regression |
| Baselines utils | scikit-learn | 1.9.1 | Splits, metrics helpers |
| Dashboard | Streamlit | 1.66.0 | Small results app; static screenshots in README |
| Test / lint | pytest / ruff | 9.1.1 / 0.17.1 | |
| CI | GitHub Actions | | lint + tests on a tiny fixture dataset |

Re-check versions at M1 before pinning in `pyproject.toml`.
Skipped: auth, payments, database server, hosting beyond optional Streamlit Community Cloud.

---

## Project Structure

```
RecRankingLab/
  PLAN.md  PROJECT.md  CLAUDE.md  README.md  Makefile  pyproject.toml
  data/            # raw/ downloaded (gitignored); fixtures/ tiny sample for tests (committed)
  sql/             # DuckDB feature + label queries, one file each
  src/recsys/      # data.py, split.py, metrics.py, baselines.py, features.py, ranker.py,
                   # twotower.py, abtest.py (power, CUPED, tests)
  app/             # Streamlit dashboard
  tests/
  docs/            # 01-evaluation-protocol.md, 02-results.md, 03-ab-simulation.md
  .github/workflows/ci.yml
```

---

## Environment Variables

None. (MLflow uses a local `mlruns/` store.)

---

## Milestones

### Milestone 1: Scaffold
**Goal:** repo runs lint and tests in a clean venv and CI.
- [x] `pyproject.toml`, Makefile (`install`, `data`, `test`, `lint`), `src/recsys/`, `tests/` — Done when: `make install && make test` passes with one placeholder test
- [x] `.github/workflows/ci.yml` running ruff + pytest — Done when: CI green on push (green on PR #1, 2026-10-21)
- [x] README stub with project goal and the "simulated A/B" disclaimer — Done when: file exists and is linked from PROJECT.md
- [x] Gate: lint and tests pass

### Milestone 2: Data and evaluation protocol
**Goal:** reproducible data load and a leakage-safe split, documented before any modelling.
- [x] `make data` downloads and unzips ml-25m into `data/raw/`, verifying the file checksum — Done when: second run is a no-op and row count is 25,000,095
- [x] `src/recsys/data.py`: load CSVs into a DuckDB file; tables `ratings`, `movies`, `tags` — Done when: `SELECT count(*)` matches expected counts (test on fixture)
- [x] `split.py`: global time cutoffs (train / val / test); per-user last-N holdout variant — Done when: tests assert no test timestamp precedes the cutoff and no user-item pair appears in two splits
- [x] Define the task in `docs/01-evaluation-protocol.md`: positive = rating ≥ 4, candidates = all unseen items, K ∈ {10, 20}, cold-start users excluded and counted — Done when: doc exists and the split code matches it
- [x] Gate: lint and tests pass

### Milestone 3: Metrics and baselines
**Goal:** trustworthy numbers for simple models.
- [ ] `metrics.py`: Recall@K, NDCG@K, MAP@K, coverage, with a bootstrap CI over users — Done when: unit tests match hand-computed values on a 5-user toy example
- [ ] `baselines.py`: global popularity, recent popularity, item-item kNN (cosine on implicit feedback) — Done when: each returns top-K per user and a test checks it excludes already-seen items
- [ ] `make eval-baselines` logs results to MLflow and writes a table to `docs/02-results.md` — Done when: table shows all three baselines on the test split with CIs
- [ ] Gate: lint and tests pass

### Milestone 4: SQL feature pipeline and LightGBM ranker
**Goal:** a learned ranker evaluated against the baselines.
- [x] `sql/*.sql`: user features (activity, mean rating, genre affinity), item features (popularity, mean rating, age, genres), user-item features (genre match, kNN score); all take an `as_of` parameter — Done when: a test builds features at a cutoff and asserts max source timestamp ≤ cutoff
- [x] Training set builder: positives + sampled negatives from the candidate set, grouped by user — Done when: group sizes sum to row count (test)
- [x] `ranker.py`: LightGBM LambdaRank trained with early stopping on val, params and metrics logged to MLflow — Done when: run appears in MLflow with val NDCG@10
- [x] Re-rank the baseline candidates on test and add to the results table; feature-importance and one ablation (drop kNN score) — Done when: `docs/02-results.md` reports the ranker vs baselines with CIs and the ablation
- [x] Gate: lint and tests pass

### Milestone 5: Two-tower candidate generation
**Goal:** replace the kNN candidate source with a learned retriever and measure the two-stage system.
- [x] `twotower.py`: user and item towers, in-batch sampled softmax, CPU training on the train split — Done when: training loss decreases on the fixture and a test checks embedding shapes
- [x] Retrieval with exact dot product (no ANN needed at this scale); report Recall@100 for candidate generation — Done when: Recall@100 beats popularity in the results table
- [x] Two-stage eval: two-tower top-200 → LightGBM re-rank, NDCG@10 vs single-stage — Done when: table updated and the MLflow registry holds both models
- [x] Gate: lint and tests pass

### Milestone 6: Simulated A/B test
**Goal:** a correct experimentation workflow on top of the models, clearly labelled as a simulation.
- [x] `docs/03-ab-simulation.md`: generative assumptions. Control = popularity ranker, treatment = LightGBM ranker. Per-user simulated engagement is drawn from held-out ratings (users randomised 50/50, outcome = whether each user engages with an item in their shown top-10) with a pre-period covariate (prior activity) — Done when: doc states every assumption and the limits
- [x] `abtest.py`: power analysis (MDE vs sample size), two-sample test with CI, CUPED adjustment, sample-ratio-mismatch check — Done when: unit tests recover a known injected effect, and CUPED variance ≤ unadjusted variance on simulated data
- [x] Simulation study: run N=1,000 A/A tests (false-positive rate ≈ α) and N=1,000 A/B tests at the injected effect (empirical power ≈ planned) — Done when: both rates are reported in the doc with the plot
- [ ] Gate: lint and tests pass

### Milestone 7: Dashboard and write-up
**Goal:** something a recruiter can open in two minutes.
- [ ] Streamlit app: metrics table, per-user recommendation viewer, A/B power curve, CUPED before/after — Done when: `make app` serves it and each tab renders without errors on the committed result artefacts
- [ ] README: problem, results table, plots, "what I'd do next", limitations (non-commercial dataset, simulated A/B), GroupLens citation — Done when: every number in the README matches `docs/02-results.md`
- [ ] Gate: lint and tests pass

### Milestone 8: Polish
- [ ] Reproduce from scratch in a clean clone (`make install data eval`) — Done when: documented command sequence works end to end
- [ ] Re-read all docs for claims not backed by a logged run — Done when: each claim links to an MLflow run or table

---

## Claude Code Commands

First session:
```
claude "Read PLAN.md and PROJECT.md. Complete Milestone 1, fetching the latest official docs for any library before using it. Update PROJECT.md. Mark tasks done as you go. Stop after Milestone 1 and commit."
```
Resume:
```
claude "Read PLAN.md and PROJECT.md. Find the first incomplete task and continue, fetching the latest official docs for any library before using it. Keep PROJECT.md in sync. Commit when a milestone is complete."
```

## Notes & Decisions
- 2026-10-21: MovieLens 25M chosen over Amazon Reviews and Instacart (user: "your call"). Public repo from the start.
- No UI brand pass: the dashboard is a small Streamlit app; keep default theme.

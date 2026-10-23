# RecRankingLab — Project Tracker

> Living context map. Keep in sync whenever structure, status, or decisions change.

_Last updated: 2026-10-21_

## What it is
A two-stage movie recommender (two-tower retrieval + LightGBM ranker) on MovieLens 25M, with a leakage-safe time split, baselines, confidence intervals, and a simulated A/B test (power analysis, CUPED). A data-science portfolio project; the point is evaluation rigor.

## Stack
Python 3.12 · DuckDB 1.5.6 · Polars 2.0.0 · LightGBM 4.7.0 · PyTorch 2.14.1 · MLflow 3.17.2 · SciPy 1.18.1 · statsmodels 0.15.0 · Streamlit 1.66.0 · pytest/ruff. Pins to be re-checked at M1.

## Architecture
raw CSV → DuckDB tables → time split → SQL features (`as_of` cutoff) → candidates (popularity / kNN / two-tower) → LightGBM LambdaRank re-rank → metrics (Recall/NDCG@K + bootstrap CI) → MLflow; results feed `docs/` and the Streamlit app; `abtest.py` simulates experiments on held-out data.

## Project structure
See PLAN.md "Project Structure". `src/recsys/{data,split}.py`, `tests/`, `docs/01-evaluation-protocol.md`; `sql/`, `app/`, `data/fixtures/` still empty. `make data` downloads, verifies and loads MovieLens into `data/movielens.duckdb` (gitignored).

## Conventions
SQL lives in `sql/`, one query per file. Every feature query takes `as_of`. Raw data is never committed. Anything simulated is labelled "simulated". No AI attribution trailers in commits.

## Current status
| Milestone | Status |
|---|---|
| M1 Scaffold | ✅ done 2026-10-21 (PR #1 merged, CI green) |
| M2 Data + eval protocol | ✅ done 2026-10-21 (branch claude/auto-2026-10-21-m2-data) |
| M3 Metrics + baselines | ✅ merged (PR #3) |
| M4 SQL features + LightGBM | ✅ merged (PR #4) |
| M5 Two-tower | ✅ done 2026-10-21 (branch claude/auto-2026-10-21-m5-twotower) |
| M6 Simulated A/B | done (PR open) |
| M7 Dashboard and write-up | done (PR open) |
| M8 Polish | — |

## Decision log
- 2026-10-21 — Dataset: MovieLens 25M. Python 3.12 + venv/pip/Makefile (uv not installed here). Public repo: github.com/ikathuria/RecRankingLab.

- 2026-10-21 — M2 protocol: positive = rating >= 4; global time split at the 80/90th timestamp percentiles; users need >= 20 train ratings; test = 3,833 eligible users (115,694 positives); 12.8% of test ratings are on items unseen in train. Details in docs/01-evaluation-protocol.md.

## Glossary
- **NDCG@K / Recall@K** — ranking quality over the top K recommendations.
- **CUPED** — variance reduction using a pre-experiment covariate.
- **MDE** — minimum detectable effect.
- **SRM** — sample ratio mismatch.

- M3 (2026-10-21): metrics.py (Recall/NDCG/MAP@K, coverage, bootstrap + paired bootstrap CI), baselines.py (popularity, recent popularity 90d, item-kNN), `make eval-baselines` -> MLflow + docs/02-results.md.
- M4 (2026-10-21): sql/*.sql feature pipeline (as_of param), features.py (candidates = union of top-100 from 3 baselines, all candidates used instead of sampled negatives), ranker.py (LightGBM LambdaRank, trained on val-period positives, features as of train end). Test Recall@20 0.0743 vs recent-pop 0.0606 (paired diff CI excludes 0). Candidate recall ceiling 0.219.
- M5 (2026-10-21): twotower.py (id + genre towers, in-batch sampled softmax with logQ, CPU, 8 epochs, ~10 min), two_stage.py in two processes (`retrieve` torch / `rank` lightgbm; mixing them segfaults on macOS). Test Recall@100 0.179 vs recent-pop 0.166 vs popularity 0.105; two-stage NDCG@10 0.1060 vs single-stage 0.0975 (paired diff +0.0085 [0.0048, 0.0121]); Recall@20 diff not significant.

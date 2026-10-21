# recRankingLab — Project Tracker

> Living context map. Keep in sync whenever structure, status, or decisions change.

_Last updated: 2026-10-21_

## What it is
A two-stage movie recommender (two-tower retrieval + LightGBM ranker) on MovieLens 25M, with a leakage-safe time split, baselines, confidence intervals, and a simulated A/B test (power analysis, CUPED). A data-science portfolio project; the point is evaluation rigor.

## Stack
Python 3.12 · DuckDB 1.5.6 · Polars 2.0.0 · LightGBM 4.7.0 · PyTorch 2.14.1 · MLflow 3.17.2 · SciPy 1.18.1 · statsmodels 0.15.0 · Streamlit 1.66.0 · pytest/ruff. Pins to be re-checked at M1.

## Architecture
raw CSV → DuckDB tables → time split → SQL features (`as_of` cutoff) → candidates (popularity / kNN / two-tower) → LightGBM LambdaRank re-rank → metrics (Recall/NDCG@K + bootstrap CI) → MLflow; results feed `docs/` and the Streamlit app; `abtest.py` simulates experiments on held-out data.

## Project structure
See PLAN.md "Project Structure". Scaffold only: `src/recsys/` (empty package), `tests/test_smoke.py`, empty `sql/ docs/ app/ data/fixtures/`.

## Conventions
SQL lives in `sql/`, one query per file. Every feature query takes `as_of`. Raw data is never committed. Anything simulated is labelled "simulated". No AI attribution trailers in commits.

## Current status
| Milestone | Status |
|---|---|
| M1 Scaffold | 🟡 done locally 2026-10-21; CI unverified until the repo is pushed |
| M2 Data + eval protocol | — |
| M3 Metrics + baselines | — |
| M4 SQL features + LightGBM | — |
| M5 Two-tower | — |
| M6 Simulated A/B | — |
| M7 Dashboard + write-up | — |
| M8 Polish | — |

## Decision log
- 2026-10-21 — Dataset: MovieLens 25M. Python 3.12 + venv/pip/Makefile (uv not installed here). Public repo planned; not yet pushed.

## Glossary
- **NDCG@K / Recall@K** — ranking quality over the top K recommendations.
- **CUPED** — variance reduction using a pre-experiment covariate.
- **MDE** — minimum detectable effect.
- **SRM** — sample ratio mismatch.

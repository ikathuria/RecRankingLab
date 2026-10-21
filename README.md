# rec-ranking-lab

A two-stage movie recommender (two-tower retrieval + LightGBM ranker) on
[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/), with a leakage-safe time split,
baselines with confidence intervals, and a **simulated** A/B test (power analysis, CUPED).

> The A/B test is a simulation built from held-out ratings under stated assumptions. It is not a
> real online experiment.

Status: scaffold only. See [PLAN.md](PLAN.md) and [PROJECT.md](PROJECT.md).

## Setup

```bash
make install
make test
```

Data: MovieLens is licensed for non-commercial research; this repo downloads it and never
redistributes it. Cite: Harper & Konstan (2015), *The MovieLens Datasets: History and Context*.

# RecRankingLab

A two-stage movie recommender (two-tower retrieval + LightGBM ranker) on
[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/), with a leakage-safe time split,
baselines with confidence intervals, and a **simulated** A/B test (power analysis, CUPED, SRM check).

> The A/B test is a simulation built from held-out ratings under stated assumptions. It is not a
> real online experiment.

## Results

Test split (ratings after 2018-01-04), 3,833 users with a positive. Mean [95% bootstrap CI over
users]. Full tables, paired differences and feature importance: [docs/02-results.md](docs/02-results.md).
Protocol: [docs/01-evaluation-protocol.md](docs/01-evaluation-protocol.md).

| system | Recall@20 | NDCG@10 |
|---|---|---|
| Popularity | 0.0385 [0.0357, 0.0419] | 0.0604 [0.0565, 0.0646] |
| Recent popularity (90d) | 0.0606 [0.0565, 0.0649] | 0.0859 [0.0808, 0.0904] |
| Item-kNN (cosine) | 0.0546 [0.0510, 0.0583] | 0.0866 [0.0817, 0.0916] |
| Single-stage LightGBM (baseline candidates) | 0.0743 [0.0699, 0.0786] | 0.0975 [0.0926, 0.1024] |
| **Two-stage: two-tower top-200 -> LightGBM** | 0.0771 [0.0730, 0.0814] | 0.1060 [0.1009, 0.1111] |

- Retrieval Recall@100: two-tower 0.1793 [0.1722, 0.1856] vs recent popularity 0.1659, item-kNN 0.1427, popularity 0.1046.
- Two-stage vs single-stage, NDCG@10: +0.0085 [0.0048, 0.0121] (paired bootstrap). On Recall@20 the
  difference is +0.0028 [-0.0011, 0.0067], so the CI includes 0: the gain is in ordering, not in
  finding more positives at 20.

### Simulated A/B test

[docs/03-ab-simulation.md](docs/03-ab-simulation.md) lists every assumption. Control = popularity
top-10, treatment = the single-stage LightGBM top-10; engagement is generated from held-out
ratings with assumed click probabilities, so the effect size is an assumption, not a forecast.

- 80% power needs 946 users per arm (modelled effect 0.059).
- 1,000 A/A tests: false-positive rate 0.042 (target 0.05). 1,000 A/B tests at 1,892 users: power
  0.787 vs 0.80 planned. 0 SRM flags.
- CUPED gave about 0.1% variance reduction: prior activity does not predict engagement here
  (rho = -0.03). Its benefit is only demonstrated on synthetic data in the tests.

![Simulated MDE vs sample size](docs/03-ab-power.png)

## Run it

```bash
make install            # needs python3.12; LightGBM needs `brew install libomp` on macOS
make data               # download + verify + load MovieLens 25M into DuckDB
make eval-baselines
.venv/bin/python -m recsys.ranker               # M4 single-stage ranker
.venv/bin/python -m recsys.two_stage all        # M5, ~25 min on CPU
.venv/bin/python -m recsys.abtest all           # M6 simulated A/B (inputs + study)
make app                # dashboard; reads only docs/ and results/
make test
```

## What I'd do next

- Train the two-tower longer (validation Recall@100 was still rising at epoch 8) and add history features to the user tower.
- Try a union of two-tower and baseline candidates to separate the effect of tt_score from the wider candidate set.
- Use a CUPED covariate that predicts engagement (pre-period positive rate).
- Ablate `i_age_years`, the top feature by gain (22.8%).

## Limitations

- MovieLens is licensed for non-commercial research; this repo downloads it and never
  redistributes it. Ratings are explicit feedback, not impressions or clicks.
- Offline metrics count only rated items as hits; good unrated items count as misses.
- 12.8% of test ratings are on items unseen in train, which caps recall.
- The A/B test is simulated (see above). One split, one seed per model.
- The dashboard's recommendation viewer shows the popularity control and the single-stage ranker
  for 200 sampled users; the two-tower stage is not exported.

## Citation

Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4).

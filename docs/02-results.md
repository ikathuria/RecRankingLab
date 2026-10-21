# Results

## Baselines on the test split (M3)

3,833 eligible users with a positive, 115,694 positives. Cells are mean [95% bootstrap CI over users, 1000 resamples, seed 0]. Coverage is the share of the 24,740 train-positive items recommended to anyone.

| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MAP@20 | coverage@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.0243 [0.0219, 0.0270] | 0.0385 [0.0357, 0.0419] | 0.0604 [0.0565, 0.0646] | 0.0587 [0.0552, 0.0625] | 0.0235 [0.0215, 0.0259] | 0.017 |
| Recent popularity (90d) | 0.0361 [0.0331, 0.0393] | 0.0606 [0.0565, 0.0649] | 0.0859 [0.0808, 0.0904] | 0.0845 [0.0799, 0.0887] | 0.0351 [0.0323, 0.0379] | 0.014 |
| Item-kNN (cosine) | 0.0347 [0.0319, 0.0377] | 0.0546 [0.0510, 0.0583] | 0.0866 [0.0817, 0.0916] | 0.0825 [0.0783, 0.0869] | 0.0356 [0.0328, 0.0385] | 0.034 |

Recall is capped below 1 because 12.8% of test ratings are on items unseen in train (docs/01-evaluation-protocol.md). Candidates: train items minus the user's train items.

## Ranker (M4)

Re-ranks the union of the top-100 from popularity, recent popularity and item-kNN (709,528 candidate rows for 3,833 test users). Candidate recall ceiling (share of a user's test positives inside the candidate set, averaged over users): 0.2187. Trained on val-period positives with features as of the end of train; early-stopped on 20% of val users. Baseline rows here are recomputed on the same users and match the table above.

| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MAP@20 | coverage@20 |
|---|---|---|---|---|---|---|
| Recent popularity (90d) | 0.0361 [0.0331, 0.0393] | 0.0606 [0.0565, 0.0649] | 0.0859 [0.0808, 0.0904] | 0.0845 [0.0799, 0.0887] | 0.0351 [0.0323, 0.0379] | 0.014 |
| Item-kNN (cosine) | 0.0347 [0.0319, 0.0377] | 0.0546 [0.0510, 0.0583] | 0.0866 [0.0817, 0.0916] | 0.0825 [0.0783, 0.0869] | 0.0356 [0.0328, 0.0385] | 0.034 |
| LightGBM ranker | 0.0465 [0.0432, 0.0499] | 0.0743 [0.0699, 0.0786] | 0.0975 [0.0926, 0.1024] | 0.0984 [0.0939, 0.1029] | 0.0406 [0.0379, 0.0433] | 0.035 |
| - without kNN score/rank | 0.0405 [0.0373, 0.0438] | 0.0661 [0.0622, 0.0703] | 0.0870 [0.0822, 0.0918] | 0.0873 [0.0831, 0.0915] | 0.0351 [0.0326, 0.0377] | 0.031 |

Paired bootstrap differences (same users resampled; mean [95% CI]):

- ranker − recent popularity, recall@20: 0.0137 [0.0099, 0.0172]
- ranker − recent popularity, ndcg@10: 0.0116 [0.0081, 0.0152]
- ranker − item-kNN, recall@20: 0.0197 [0.0159, 0.0239]
- ranker − item-kNN, ndcg@10: 0.0109 [0.0060, 0.0154]
- ranker − ranker without kNN, recall@20: 0.0082 [0.0056, 0.0111]
- ranker − ranker without kNN, ndcg@10: 0.0105 [0.0083, 0.0127]

Feature importance (gain share, full model):

| feature | share |
|---|---|
| i_age_years | 22.8% |
| knn_rank | 15.6% |
| i_pos_90d | 13.6% |
| genre_match | 9.1% |
| knn_score | 6.1% |
| recent_rank | 5.5% |
| i_n_pos | 3.6% |
| i_n_ratings | 3.4% |
| u_span_days | 3.3% |
| u_days_since_last | 3.3% |
| i_mean_rating | 2.8% |
| u_n_ratings | 2.3% |
| pop_rank | 2.1% |
| item_vs_user_mean | 1.6% |
| u_pos_rate | 1.5% |
| u_mean_rating | 1.1% |
| i_n_genres | 1.1% |
| i_pos_rate | 1.1% |

## Two-tower retrieval and two-stage ranking (M5)

Two-tower: user-id and item-id(+genre) towers, 64-d L2-normalised embeddings, in-batch sampled softmax with logQ correction, temperature 0.05, batch 4096, Adam 2e-3, 137,682 users x 24,740 items, 9,981,093 train positives; trained in 9.6 min on CPU, epoch 8 kept (best val Recall@100 on 4,948 val users). Retrieval over all items is an exact dot product (0 s for 3,833 test users). Items the user rated in train are excluded for every retriever.

### Retrieval on the test split

| retriever | Recall@100 | Recall@200 |
|---|---|---|
| Two-tower (exact dot product) | 0.1793 [0.1722, 0.1856] | 0.2510 [0.2434, 0.2584] |
| Popularity | 0.1046 [0.0991, 0.1101] | 0.1481 [0.1417, 0.1541] |
| Recent popularity (90d) | 0.1659 [0.1594, 0.1725] | 0.2320 [0.2244, 0.2395] |
| Item-kNN (cosine) | 0.1427 [0.1368, 0.1492] | 0.2008 [0.1939, 0.2078] |

Paired bootstrap differences (mean [95% CI]):

- two-tower − Popularity, recall@100: 0.0747 [0.0690, 0.0798]
- two-tower − Popularity, recall@200: 0.1029 [0.0968, 0.1085]
- two-tower − Recent popularity (90d), recall@100: 0.0133 [0.0071, 0.0193]
- two-tower − Recent popularity (90d), recall@200: 0.0190 [0.0126, 0.0251]
- two-tower − Item-kNN (cosine), recall@100: 0.0366 [0.0321, 0.0407]
- two-tower − Item-kNN (cosine), recall@200: 0.0502 [0.0457, 0.0546]

Training history:

| epoch | train loss | val Recall@100 |
|---|---|---|
| 1 | 7.780 | 0.2400 |
| 2 | 7.394 | 0.2523 |
| 3 | 7.304 | 0.2579 |
| 4 | 7.246 | 0.2667 |
| 5 | 7.200 | 0.2633 |
| 6 | 7.161 | 0.2686 |
| 7 | 7.130 | 0.2695 |
| 8 | 7.105 | 0.2707 |

### Two-stage vs single-stage (3,833 test users)

Both rankers use the same features (the two-stage one adds tt_score and tt_rank), LightGBM parameters and training protocol as M4. The candidate recall ceiling is the share of a user's test positives inside the candidate set.

| system | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | candidate recall ceiling |
|---|---|---|---|---|---|
| Two-tower alone (top-20 by dot product) | 0.0403 [0.0376, 0.0435] | 0.0665 [0.0624, 0.0706] | 0.0980 [0.0928, 0.1030] | 0.0959 [0.0913, 0.1004] |  |
| Single-stage LightGBM (baseline candidates) | 0.0465 [0.0432, 0.0499] | 0.0743 [0.0699, 0.0786] | 0.0975 [0.0926, 0.1024] | 0.0984 [0.0939, 0.1029] | 0.2187 |
| Two-stage: two-tower top-200 -> LightGBM | 0.0473 [0.0442, 0.0506] | 0.0771 [0.0730, 0.0814] | 0.1060 [0.1009, 0.1111] | 0.1056 [0.1010, 0.1104] | 0.2510 |

Paired bootstrap differences:

- two-stage − single-stage, ndcg@10: 0.0085 [0.0048, 0.0121]
- two-stage − single-stage, ndcg@20: 0.0072 [0.0042, 0.0104]
- two-stage − single-stage, recall@10: 0.0008 [-0.0025, 0.0041]
- two-stage − single-stage, recall@20: 0.0028 [-0.0011, 0.0067]

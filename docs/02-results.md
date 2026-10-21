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

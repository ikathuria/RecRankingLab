# Results

## Baselines on the test split (M3)

3,833 eligible users with a positive, 115,694 positives. Cells are mean [95% bootstrap CI over users, 1000 resamples, seed 0]. Coverage is the share of the 24,740 train-positive items recommended to anyone.

| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MAP@20 | coverage@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.0243 [0.0219, 0.0270] | 0.0385 [0.0357, 0.0419] | 0.0604 [0.0565, 0.0646] | 0.0587 [0.0552, 0.0625] | 0.0235 [0.0215, 0.0259] | 0.017 |
| Recent popularity (90d) | 0.0361 [0.0331, 0.0393] | 0.0606 [0.0565, 0.0649] | 0.0859 [0.0808, 0.0904] | 0.0845 [0.0799, 0.0887] | 0.0351 [0.0323, 0.0379] | 0.014 |
| Item-kNN (cosine) | 0.0347 [0.0319, 0.0377] | 0.0546 [0.0510, 0.0583] | 0.0866 [0.0817, 0.0916] | 0.0825 [0.0783, 0.0869] | 0.0356 [0.0328, 0.0385] | 0.034 |

Recall is capped below 1 because 12.8% of test ratings are on items unseen in train (docs/01-evaluation-protocol.md). Candidates: train items minus the user's train items.

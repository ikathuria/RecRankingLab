# Evaluation protocol

Fixed before any modelling. Code: `src/recsys/split.py`; tests: `tests/test_split.py`.

## Task
For each evaluated user, rank all items the user has not interacted with before the cutoff and
measure how many of the user's later positives appear in the top K.

- **Positive:** rating >= 4.0.
- **Metrics:** Recall@K and NDCG@K (K = 10, 20), MAP@K, catalogue coverage (milestone 3), with
  bootstrap confidence intervals over users.

## Split (global time split)
Cutoffs are the 80th and 90th percentile of rating timestamps over the full dataset.

| split | rule | ratings | users | period |
|---|---|---|---|---|
| train | ts <= 2016-06-25 | 20,000,076 | 137,883 | 1995-01 to 2016-06 |
| val | up to 2018-01-04 | 2,500,010 | 18,003 | |
| test | after 2018-01-04 | 2,500,009 | 18,442 | to 2019-11 |

Test data always comes from after every training rating, so nothing about the future leaks into
training or features. A (user, item) pair occurs once in MovieLens, so no pair is in two splits
(tested).

## Who is evaluated
A user is **eligible** if they have >= 20 ratings in train. Others are cold-start users: excluded
from evaluation and counted.

- Eligible users: 137,567 of 162,541.
- Cold-start users with a positive in val/test: 24,961 (excluded).
- Eligible users with at least one positive: **val 4,948 users / 159,929 positives; test 3,833
  users / 115,694 positives.**

## Things to know before reading results
- **Few users, many positives each.** Most MovieLens users rate in a single burst, so only ~3% of
  eligible users have later activity. Test results are over 3,833 users, which is why bootstrap
  CIs matter.
- **Cold items.** 12.8% of test ratings (318,980 of 2,500,009) are on items never rated in train.
  Models that only see train items cannot recommend them; this caps achievable recall and applies
  equally to every model.
- **Candidates:** all items seen in train, minus the items the user already rated in train.
- **Scope of the claim:** offline accuracy on historical ratings, not online behaviour.

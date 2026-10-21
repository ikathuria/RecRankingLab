# Handoff: 2026-10-21 · M3 + M4 done (unmerged), M5 next

**Goal:** Portfolio data-science project: two-stage MovieLens 25M recommender with leakage-safe eval and a simulated A/B test. See PLAN.md (8 milestones).
**Status:** M1, M2 merged. M3 = PR #3 (CI green, **not merged**: harness blocked the merge). M4 = PR #4, stacked on #3 (CI not yet checked). Working branch `claude/auto-2026-10-21-m4-ranker`.

## Next step
1. Merge #3 (`gh pr merge 3 --merge`; owner may need to run it with `!`), retarget #4 to `main`, check CI, merge #4, `git checkout main && git pull`.
2. **M5**: `src/recsys/twotower.py` (user/item towers, in-batch sampled softmax, CPU), exact dot-product retrieval, Recall@100 vs popularity, then two-stage eval (two-tower top-200 -> LightGBM re-rank). Branch from main after the merges.
   -> verify: loss decreases on fixture, embedding-shape test, table in `docs/02-results.md`.

## Done
- M3: `metrics.py`, `baselines.py`, `eval_baselines.py` (`make eval-baselines`), 13 tests (ef88a67).
- M4: `sql/*.sql` (as_of), `features.py`, `ranker.py`, `tests/test_features.py` (98991dc). 23 tests pass, lint clean.
- Test split results (3,833 users): Recall@20 recent-pop 0.0606, item-kNN 0.0546, LightGBM 0.0743 [0.0699, 0.0786]; paired diff vs recent-pop +0.0137 [0.0099, 0.0172]; dropping kNN features -0.0082 Recall@20. Details in `docs/02-results.md`.

## Decisions
- Ranker features always as of train end; trained on val-period positives, early-stopped on 20% of val users; does not see val ratings at test time (fair vs baselines).
- Candidates = union of top-100 from popularity, recent popularity, item-kNN; all candidates used (no negative sampling, deviation from PLAN).
- Ablation drops both `knn_score` and `knn_rank`.

## Open threads
- `i_age_years` is the top feature (22.8% gain). [Likely] partly a val-period "recent releases" artifact; untested. A quick ablation would settle it.
- Candidate recall ceiling is only 0.219: caps any re-ranker, and is the motivation for M5 retrieval.
- Recent popularity (90d window before train end) is the real bar for any model.

## Gotchas
- LightGBM on macOS needs `brew install libomp` (done here).
- Agent `git push` worked this session; `gh pr merge` was denied by the classifier unless explicitly requested per PR.
- MLflow logs to local `mlflow.db`/`mlruns` (gitignored). Metric names can't contain `@`.
- Dataset builds cached in `data/cache/*.parquet` (gitignored); delete when features change.
- `make eval-baselines` ~95 s; `python -m recsys.ranker` ~6 min.
- Earlier: moving the repo breaks `.venv` shebangs (`rm -rf .venv && make install`); `timeout`/`uv` not installed.

## Read first
- `PLAN.md`: milestones. `docs/01-evaluation-protocol.md`: split and candidates. `docs/02-results.md`: numbers.
- `src/recsys/features.py`, `src/recsys/ranker.py`, `src/recsys/baselines.py` (`KnnScorer`).

## Uncommitted
clean after this file is committed

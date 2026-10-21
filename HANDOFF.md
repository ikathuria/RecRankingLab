# Handoff: 2026-10-21 · M1-M5 done, M6 next

**Goal:** Portfolio data-science project: two-stage MovieLens 25M recommender with leakage-safe eval and a simulated A/B test. See PLAN.md (8 milestones).
**Status:** M1-M4 merged to `main`. M5 on branch `claude/auto-2026-10-21-m5-twotower` (PR open, not merged).

## Next step
Check CI on PR #5 (was pending at handoff) and merge it, then **M6** (`abtest.py`: simulated A/B test, power, CUPED, SRM; see PLAN.md). Label everything simulated.

## Done (M5)
- `src/recsys/twotower.py`, `src/recsys/two_stage.py` (`python -m recsys.two_stage all`, ~25 min), `build_candidates(retrieved=...)` in `features.py`, 5 new tests (28 pass, lint clean).
- Test split (3,833 users): Recall@100 two-tower 0.1793 [0.1722, 0.1856], recent-pop 0.1659, item-kNN 0.1427, popularity 0.1046; Recall@200 0.2510 vs 0.2320 recent-pop.
- Two-stage NDCG@10 0.1060 vs single-stage 0.0975 (paired +0.0085 [0.0048, 0.0121]); Recall@20 0.0771 vs 0.0743, diff CI includes 0. Two-tower alone NDCG@10 0.0980 is about equal to the single-stage ranker. Details in `docs/02-results.md`.
- MLflow registry (local `mlflow.db`): two-tower-retriever, single-stage-ranker, two-stage-ranker, version 1 each.

## Decisions
- User tower is a user-id embedding (no history features); item tower is item id + genres. Eligible users are all train users, so ids are seen.
- Epoch chosen by val Recall@100; training stopped at 8 epochs.
- Two-stage ranker trained exactly like M4 (val-period positives, features as of train end) on two-tower top-200 candidates, plus tt_score/tt_rank.
- Torch model logged with `serialization_format="pickle"` (pt2 needs an input example); pickle loads can run code, only load your own.

## Open threads
- Val Recall@100 was still rising at epoch 8 (0.2707); [Likely] more epochs help a little. Not run.
- The gain is on NDCG, not Recall@20; [Guessing] it comes from tt_score ordering rather than the wider candidate set. A union of two-tower + baseline candidates was not tried.
- `i_age_years` ablation from M4 still untested.

## Gotchas
- **torch and lightgbm in one process segfault on macOS** (libomp clash at LightGBM Dataset creation). `two_stage` runs `retrieve` (torch) and `rank` (lightgbm) as separate processes sharing `data/cache/two_stage_all.pkl`; keep `import torch` out of the rank path. `ruff --fix` deletes the guarded `import torch` unless it keeps its noqa.
- Earlier: LightGBM needs `brew install libomp`; MLflow metric names can't contain `@`; dataset caches in `data/cache` (delete when features change); moving the repo breaks `.venv`; `timeout`/`uv` not installed.

## Read first
`PLAN.md`, `docs/02-results.md`, `src/recsys/two_stage.py`, `src/recsys/twotower.py`.

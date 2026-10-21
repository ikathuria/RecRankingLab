# Handoff: 2026-10-21 · M2 data + split done, M3 next

**Goal:** Portfolio data-science project (JobApply PLAN M43): two-stage MovieLens 25M recommender with a leakage-safe eval and a simulated A/B test. See PLAN.md (8 milestones).
**Status:** M1 merged (PR #1, CI green). M2 done on local branch `claude/auto-2026-10-21-m2-data` (`f94221f`), **not pushed, no PR yet**.

## Next step
1. Push M2 and open the PR (the harness blocks pushes from the agent, so the owner runs it): `git push -u origin claude/auto-2026-10-21-m2-data && gh pr create -B main -t "M2: data and split" -f`. Confirm CI green, merge, `git checkout main && git pull`.
2. Start **M3** (metrics + baselines): `src/recsys/metrics.py` (Recall/NDCG/MAP@K, coverage, bootstrap CI over users) with hand-computed unit tests; then `baselines.py` (popularity, recent popularity, item-kNN).
   → verify: tests pass; baseline table on the test split in `docs/02-results.md`.

## Done
- M1: pyproject, Makefile (`install/test/lint/data`), CI, smoke test, README (with "simulated A/B" disclaimer).
- M2: `data.py` (download + md5 check + DuckDB load; 25,000,095 ratings verified), `split.py` (global time split 80/90th percentile; eligible = >=20 train ratings), 6 tests, `docs/01-evaluation-protocol.md`.

## Decisions
- MovieLens 25M (over Amazon Reviews, Instacart); public repo `ikathuria/RecRankingLab`; Python 3.12 + venv/pip/Makefile (no uv here); package named `recsys`.
- Positive = rating >= 4; K = 10, 20; candidates = train items minus the user's train items.

## Findings that shape later work
- Test = only 3,833 eligible users (115,694 positives): most users rate in one burst. Use bootstrap CIs; don't over-read small gaps.
- 12.8% of test ratings are on items unseen in train: caps recall for every model. Say so in results.

## Gotchas
- The agent cannot `git push` / `gh repo create` (classifier denies); the owner runs them with `!`. Long commands wrap in the terminal and break: keep them on one line.
- Moving/renaming the repo breaks `.venv` shebangs: `rm -rf .venv && make install`.
- `data/raw/` (260 MB zip) and `data/movielens.duckdb` (490 MB) are gitignored; `make data` recreates them in ~2 s once the zip exists.
- `timeout` and `uv` are not installed.

## Read first
`PLAN.md` (milestones), `PROJECT.md` (status, decision log), `docs/01-evaluation-protocol.md`, `src/recsys/split.py`, `src/recsys/data.py`.

## Uncommitted
clean (HANDOFF.md committed with this handoff)

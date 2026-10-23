PY := .venv/bin/python

.PHONY: install test lint data eval-baselines app

install:
	python3.12 -m venv .venv
	$(PY) -m pip install -q -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

# Download + verify (md5) + extract MovieLens 25M, then load into data/movielens.duckdb
data:
	$(PY) -m recsys.data all

eval-baselines:
	$(PY) -m recsys.eval_baselines --split test

app:
	.venv/bin/streamlit run app/streamlit_app.py

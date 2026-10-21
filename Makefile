PY := .venv/bin/python

.PHONY: install test lint

install:
	python3.12 -m venv .venv
	$(PY) -m pip install -q -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

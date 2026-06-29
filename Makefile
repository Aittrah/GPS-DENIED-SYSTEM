.PHONY: install-dev test format lint typecheck check

install-dev:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -q

format:
	python3 -m black src tests simulation/scripts

lint:
	python3 -m ruff check src tests simulation/scripts

typecheck:
	python3 -m mypy src/vns

check:
	python3 -m pytest -q
	python3 -m ruff check src tests simulation/scripts
	python3 -m mypy src/vns

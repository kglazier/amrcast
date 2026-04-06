.PHONY: install install-dev install-gpu test lint format typecheck clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]"

install-all:
	pip install -e ".[dev,gpu,api]"

test:
	pytest tests/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/amrcast/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: help lint format check-docs test test-native build clean

# Python sources that the linters and docstring checkers police.
PYSRC = purc/

help:
	@echo "Available commands:"
	@echo "  make build       - Editable install with the native core (no build isolation)"
	@echo "  make lint        - Run all linters (ruff + pydocstyle + darglint)"
	@echo "  make format      - Format code with ruff"
	@echo "  make check-docs  - Check docstring style with pydocstyle"
	@echo "  make test        - Run the Python test suite"
	@echo "  make test-native - Build the native core and run its parity tests"
	@echo "  make clean       - Clean build artifacts and caches"

build:
	pip install --no-build-isolation -e .

lint:
	@echo "Running ruff (code quality)..."
	ruff check $(PYSRC) tests/ scripts/
	@echo "Running pydocstyle (Google-style docstrings)..."
	pydocstyle $(PYSRC)
	@echo "Running darglint (docstring/signature consistency)..."
	darglint --docstring-style google --strictness short $(PYSRC) || true

format:
	ruff format $(PYSRC) tests/ scripts/
	ruff check --fix $(PYSRC) tests/ scripts/

check-docs:
	pydocstyle $(PYSRC)
	darglint --docstring-style google --strictness short $(PYSRC) || true

test:
	pytest tests/ -v

test-native:
	pytest tests/ -v -m native

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/

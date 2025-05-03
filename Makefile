.PHONY: lint format test clean setup-pre-commit install install-dev

# Variables
UV = uv
SRC_DIR = src

# Code linting
lint:
	$(UV) run ruff check $(SRC_DIR)
	$(UV) run pyright $(SRC_DIR)

# Automatic code formatting
format:
	$(UV) run ruff format $(SRC_DIR)

# Run tests
test:
	@echo "Tests will be implemented later"
	# Uncomment the command below when you add tests
	# $(UV) run pytest $(SRC_DIR)/tests

# Install project dependencies
install:
	$(UV) pip install .

# Install project in development mode
install-dev:
	$(UV) pip install -e .
	$(UV) run pre-commit install --install-hooks

# Clean cache and artifacts
clean:
	rm -rf .ruff_cache
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make lint             - Check code with linters (ruff, pyright)"
	@echo "  make format           - Format code with ruff"
	@echo "  make test             - Run tests"
	@echo "  make install          - Install the package"
	@echo "  make install-dev      - Install the package in development mode"
	@echo "  make clean            - Remove cache and temporary files"

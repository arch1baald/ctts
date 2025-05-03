.PHONY: lint format test clean

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

# Clean cache and artifacts
clean:
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make lint      - Check code with linters (ruff, pyright)"
	@echo "  make format    - Format code with ruff"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Remove cache and temporary files" 
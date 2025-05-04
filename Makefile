# Variables
UV = uv
SRC_DIR = src

# Check if uv is installed
ifeq ($(shell which $(UV) 2>/dev/null),)
$(error "uv utility not found. Please install it: https://docs.astral.sh/uv/getting-started/installation/")
endif

# Code linting
.PHONY: lint
lint:
	$(UV) run ruff format $(SRC_DIR)
	$(UV) run ruff check $(SRC_DIR) --fix
	$(UV) run pyright $(SRC_DIR)

# Run tests
.PHONY: test
test:
	@echo "Tests will be implemented later"
	# Uncomment the command below when you add tests
	# $(UV) run pytest $(SRC_DIR)/tests

# Install project dependencies
.PHONY: install
install:
	$(UV) pip install .

# Install project in development mode
.PHONY: install-dev
install-dev:
	$(UV) pip install -e .
	$(UV) run pre-commit install --install-hooks

# Clean cache and artifacts
.PHONY: clean
clean:
	rm -rf .ruff_cache
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Show help
help:
	@echo "Available commands:"
	@echo "  make lint             - Check code with linters (ruff, pyright)"
	@echo "  make test             - Run tests"
	@echo "  make install          - Install the package"
	@echo "  make install-dev      - Install the package in development mode"
	@echo "  make clean            - Remove cache and temporary files"

.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: test
test:
	uv run pytest
	uv run pyright

.PHONY: build
build:
	uv build
	uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
	uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
	@echo "Build and import test successful"

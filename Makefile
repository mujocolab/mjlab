.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest

.PHONY: test-all
test-all: check test

.PHONY: build
build:
	uv build
	uv run --isolated --no-project --with dist/*.whl --with git+https://github.com/google-deepmind/mujoco_warp tests/smoke_test.py
	uv run --isolated --no-project --with dist/*.tar.gz --with git+https://github.com/google-deepmind/mujoco_warp tests/smoke_test.py
	@echo "Build and import test successful"

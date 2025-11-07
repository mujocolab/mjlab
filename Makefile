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

.PHONY: test-fast
test-fast:
	uv run pytest -m "not slow"

.PHONY: test-cpu
test-cpu:
	FORCE_CPU=1 uv run pytest

.PHONY: test-cpu-fast
test-cpu-fast:
	FORCE_CPU=1 uv run pytest -m "not slow"

.PHONY: test-all
test-all: check test

.PHONY: build
build:
	uv build
	uv run --isolated --no-project --with dist/*.whl --with git+https://github.com/google-deepmind/mujoco_warp tests/smoke_test.py
	uv run --isolated --no-project --with dist/*.tar.gz --with git+https://github.com/google-deepmind/mujoco_warp tests/smoke_test.py
	@echo "Build and import test successful"

.PHONY: docker-build
docker-build:
	docker build -t mjlab:latest .

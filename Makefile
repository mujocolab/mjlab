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
	uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
	uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
	@echo "Build and import test successful"

.PHONY: docs
docs:
	uv run --group docs sphinx-build docs docs/_build

.PHONY: docs-watch
docs-watch:
	uv run --group docs sphinx-autobuild docs docs/_build

.PHONY: publish-test
publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/

.PHONY: publish
publish: build
	uv publish

.PHONY: docker-build
docker-build:
	docker build -t mjlab:latest .

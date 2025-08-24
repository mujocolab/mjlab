SHELL := /bin/bash
.DEFAULT_GOAL := help

CYAN := \033[0;36m
GREEN := \033[0;32m
NC := \033[0m

.PHONY: help
help: ## display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n${CYAN}Usage:${NC}\n  make ${GREEN}<target>${NC}\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  ${GREEN}%-15s${NC} %s\n", $$1, $$2 } /^##@/ { printf "\n${CYAN}%s${NC}\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: sync
sync: ## sync dependencies with uv
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format: ## ruff format and lint
	uv run ruff format
	uv run ruff check --fix

.PHONY: test
test: ## run tests with pytest
	uv run pytest 
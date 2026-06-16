# Convenience targets for local development and the end-to-end dry run.
# Everything runs through `uv` so the environment is reproducible.

.PHONY: install lint test sync-data paper-step research validate report e2e clean

install:
	uv sync

lint:
	uv run ruff check engine tests
	uv run ruff format --check engine tests

fmt:
	uv run ruff format engine tests
	uv run ruff check --fix engine tests

test:
	uv run pytest

sync-data:
	uv run trade sync-data

paper-step:
	uv run trade paper-step

research:
	uv run trade research

validate:
	uv run trade validate

report:
	uv run trade report

# Local end-to-end smoke (mirrors the milestone "ローカルE2E" check).
e2e: sync-data paper-step report

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/cache

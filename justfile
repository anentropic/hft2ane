default:
    @just --list

# Run linter
lint:
    uv run ruff check hft2ane tests

# Auto-fix lint + format
fix:
    uv run ruff check --fix hft2ane tests
    uv run ruff format hft2ane tests

# Run type checker
typecheck:
    uv run basedpyright

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

# Run all quality checks
check: lint typecheck test

# Run pre-commit on all files
pre-commit:
    prek run --all-files

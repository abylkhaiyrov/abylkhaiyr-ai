#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/ruff check src tests scripts
.venv/bin/black --check -q src tests scripts
.venv/bin/mypy src
.venv/bin/pytest

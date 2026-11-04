#!/usr/bin/env bash
# Create .venv (Python 3.13), install the pinned dependencies and this package,
# then run the test suite.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3.13}"
if command -v uv >/dev/null; then
  uv venv --python 3.13 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
  uv pip install --python .venv/bin/python --no-deps -e .
else
  "$PYTHON" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip install --no-deps -e .
fi
.venv/bin/python -m pytest -q

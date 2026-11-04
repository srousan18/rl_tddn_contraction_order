# Sourced by the scripts: run from the repository root with the project's virtual environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-.venv/bin/python}"
export PYTHONPATH=src

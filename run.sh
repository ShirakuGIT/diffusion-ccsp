#!/usr/bin/env zsh
# Usage: ./run.sh [args passed to runner.py]
# Example: ./run.sh --suite graph_score_two_phase

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

conda activate diffusion-ccsp 2>/dev/null || true

export PYTHONPATH="${REPO_DIR}/packing_models:${PYTHONPATH}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export PYTHONPATH="${REPO_DIR}/../Jacinle:${PYTHONPATH}"

exec python "${REPO_DIR}/experiments/constraint_composition/runner.py" "$@"

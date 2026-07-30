#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="$(command -v uv || true)"

if [[ -z "${UV_BIN}" ]]; then
  echo "Missing uv. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

"${UV_BIN}" lock --project "${PROJECT_ROOT}" --check
"${UV_BIN}" run --project "${PROJECT_ROOT}" python -m unittest discover \
  -s "${PROJECT_ROOT}/tests" -v
"${UV_BIN}" pip check --python "${PROJECT_ROOT}/.venv/bin/python"
bash -n "${PROJECT_ROOT}/scripts/run_all.sh"
bash -n "${PROJECT_ROOT}/scripts/download_notebooks.sh"
bash -n "${PROJECT_ROOT}/scripts/download_leaderboard_notebooks.sh"
bash -n "${PROJECT_ROOT}/scripts/download_top_voted_givemesomecredit.sh"
bash -n "${PROJECT_ROOT}/scripts/download_top_voted_other_competitions.sh"
bash -n "${PROJECT_ROOT}/scripts/pull_from_tho2.sh"
bash -n "${PROJECT_ROOT}/scripts/push_to_tho2.sh"
"${UV_BIN}" run --project "${PROJECT_ROOT}" python -m compileall -q \
  "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/scripts" "${PROJECT_ROOT}/tests"

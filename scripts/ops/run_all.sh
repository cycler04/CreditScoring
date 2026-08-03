#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="$(command -v uv || true)"

if [[ -z "${UV_BIN}" ]]; then
  echo "Missing uv. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

"${UV_BIN}" run --project "${PROJECT_ROOT}" python \
  "${PROJECT_ROOT}/scripts/data/download_data.py"
"${UV_BIN}" run --project "${PROJECT_ROOT}" python \
  "${PROJECT_ROOT}/scripts/pipelines/run_pipeline.py"

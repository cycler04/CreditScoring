#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="$(command -v uv || true)"

if [[ -z "${UV_BIN}" ]]; then
  echo "Missing uv. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

NOTEBOOK_DIRS=(
  "01-start-here-a-gentle-introduction"
  "02-home-credit-complete-eda-feature-importance"
  "03-credit-risk-eda-defaults-segments-trends"
  "04-credit-risk-eda-woe-scorecard"
)

KAGGLE_KERNELS=(
  "willkoehrsen/start-here-a-gentle-introduction"
  "codename007/home-credit-complete-eda-feature-importance"
  "beatafaron/credit-risk-eda-defaults-segments-trends-1"
  "beatafaron/credit-risk-eda-woe-scorecard-2"
)

for index in "${!KAGGLE_KERNELS[@]}"; do
  target="${PROJECT_ROOT}/notebooks/${NOTEBOOK_DIRS[$index]}"
  mkdir -p "${target}"
  "${UV_BIN}" run --project "${PROJECT_ROOT}" kaggle kernels pull \
    "${KAGGLE_KERNELS[$index]}" \
    --path "${target}" \
    --metadata
done

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="$(command -v uv || true)"

if [[ -z "${UV_BIN}" ]]; then
  echo "Missing uv. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

TARGET_DIRS=(
  "home-credit-default-risk/01-home-aloan/01-xgb-simple-features"
  "home-credit-default-risk/01-home-aloan/02-lighgbm-with-selected-features"
  "home-credit-default-risk/01-home-aloan/03-good-fun-with-lightgbm"
  "home-credit-default-risk/02-ikiri-ds/01-giba-user-id-boost"
  "home-credit-model-stability/01-yuuniee/01-lightgbm"
  "home-credit-model-stability/01-yuuniee/02-catboost-inference"
  "home-credit-model-stability/01-yuuniee/03-lightautoml-inference"
)

KAGGLE_KERNELS=(
  "tunguz/xgb-simple-features"
  "ogrellier/lighgbm-with-selected-features"
  "ogrellier/good-fun-with-ligthgbm"
  "titericz/giba-post-processing-user-id-boost"
  "yuuniekiri/fork-of-home-credit-risk-lightgbm"
  "yuuniekiri/fork-of-home-credit-catboost-inference"
  "yuuniekiri/fork-of-home-credit-lightautoml-inference"
)

for index in "${!KAGGLE_KERNELS[@]}"; do
  target="${PROJECT_ROOT}/notebooks/leaderboard/${TARGET_DIRS[$index]}"
  mkdir -p "${target}"
  "${UV_BIN}" run --project "${PROJECT_ROOT}" kaggle kernels pull \
    "${KAGGLE_KERNELS[$index]}" \
    --path "${target}" \
    --metadata
done

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="$(command -v uv || true)"

if [[ -z "${UV_BIN}" ]]; then
  echo "Missing uv. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

TARGET_DIRS=(
  "home-credit-default-risk/01-gentle-introduction"
  "home-credit-default-risk/02-complete-eda-feature-importance"
  "home-credit-default-risk/03-lightgbm-simple-features"
  "home-credit-default-risk/04-manual-feature-engineering"
  "home-credit-default-risk/05-null-importances"
  "home-credit-default-risk/06-model-tuning"
  "home-credit-default-risk/07-automated-feature-engineering"
  "home-credit-default-risk/08-extensive-eda"
  "home-credit-default-risk/09-feature-selection"
  "home-credit-default-risk/10-good-fun-lightgbm"
  "home-credit-model-stability/01-starter-notebook"
  "home-credit-model-stability/02-baseline"
  "home-credit-model-stability/03-lb-567"
  "home-credit-model-stability/04-lightgbm-catboost"
  "home-credit-model-stability/05-utility-scripts"
  "home-credit-model-stability/06-this-is-the-way"
  "home-credit-model-stability/07-fork-this-is-the-way"
  "home-credit-model-stability/08-starter-inference"
  "home-credit-model-stability/09-metric-hack"
  "home-credit-model-stability/10-lgb-cat-ensemble"
)

KAGGLE_KERNELS=(
  "willkoehrsen/start-here-a-gentle-introduction"
  "codename007/home-credit-complete-eda-feature-importance"
  "jsaguiar/lightgbm-with-simple-features"
  "willkoehrsen/introduction-to-manual-feature-engineering"
  "ogrellier/feature-selection-with-null-importances"
  "willkoehrsen/intro-to-model-tuning-grid-and-random-search"
  "willkoehrsen/automated-feature-engineering-basics"
  "gpreda/home-credit-default-risk-extensive-eda"
  "willkoehrsen/introduction-to-feature-selection"
  "ogrellier/good-fun-with-ligthgbm"
  "jetakow/home-credit-2024-starter-notebook"
  "greysky/home-credit-baseline"
  "hideyukizushi/home-aftersubmissionsopen-3-11-2024-lb-567"
  "pereradulina/credit-risk-prediction-with-lightgbm-and-catboost"
  "batprem/home-credit-risk-mode-utility-scripts"
  "tritionalval/this-is-the-way"
  "hngbimnh/fork-of-this-is-the-way"
  "ravi20076/homecredit-starter-inference-v1"
  "andreasbis/metric-hack-implementation"
  "jeffersusus/home-credit-lgb-cat-ensemble"
)

for index in "${!KAGGLE_KERNELS[@]}"; do
  target="${PROJECT_ROOT}/notebooks/top-voted/${TARGET_DIRS[$index]}"
  mkdir -p "${target}"
  "${UV_BIN}" run --project "${PROJECT_ROOT}" kaggle kernels pull \
    "${KAGGLE_KERNELS[$index]}" \
    --path "${target}" \
    --metadata
done

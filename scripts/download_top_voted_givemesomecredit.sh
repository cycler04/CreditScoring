#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="$(command -v uv || true)"

if [[ -z "${UV_BIN}" ]]; then
  echo "Missing uv. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

TARGET_DIRS=(
  "01-credit-scorecard-example"
  "02-starter-credit-card-scoring"
  "03-comp-stats-group-project"
  "04-modeling-give-me-some-credit"
  "05-eda-top-100-leaderboard"
  "06-credit-top5-solution-evaluation"
  "07-eda-xgboost-lightgbm-shap"
  "08-financial-distress-prediction"
  "09-mljar-automl"
  "10-starter-give-me-some-credit"
)

KAGGLE_KERNELS=(
  "orange90/credit-scorecard-example"
  "riteshrhyme/starter-credit-card-scoring-bbe98584-0"
  "simonpfish/comp-stats-group-data-project-final"
  "caesarlupum/modeling-give-me-some-credit"
  "nicholasgah/eda-credit-scoring-top-100-on-leaderboard"
  "bannourchaker/credit-top5-solution-evaluation-all"
  "uditnagar5/give-me-some-credit-eda-xgboost-lightgbm-shap"
  "prasadposture121/financial-distress-prediction"
  "mt77pp/mljar-automl-givemesomecredit"
  "mostig/starter-give-me-some-credit"
)

for index in "${!KAGGLE_KERNELS[@]}"; do
  target="${PROJECT_ROOT}/notebooks/top-voted/GiveMeSomeCredit/${TARGET_DIRS[$index]}"
  mkdir -p "${target}"
  "${UV_BIN}" run --project "${PROJECT_ROOT}" kaggle kernels pull \
    "${KAGGLE_KERNELS[$index]}" \
    --path "${target}" \
    --metadata
done

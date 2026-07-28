# GiveMeSomeCredit — top-voted Code notebooks

Snapshot of the top 10 entries returned by the Kaggle competition Code tab,
sorted by vote count on 2026-07-28. These are community notebooks, not the
original 2011 leaderboard-winning code.

| Rank | Votes | Notebook |
|---:|---:|---|
| 1 | 336 | [Credit ScoreCard example](01-credit-scorecard-example/credit-scorecard-example.ipynb) |
| 2 | 233 | [Starter: credit card scoring](02-starter-credit-card-scoring/starter-credit-card-scoring-bbe98584-0.ipynb) |
| 3 | 205 | [Comp Stats Group Data Project](03-comp-stats-group-project/comp-stats-group-data-project-final.ipynb) |
| 4 | 199 | [Modeling: Give Me Some Credit](04-modeling-give-me-some-credit/modeling-give-me-some-credit.ipynb) |
| 5 | 107 | [EDA — Top 100 on Leaderboard](05-eda-top-100-leaderboard/eda-credit-scoring-top-100-on-leaderboard.ipynb) |
| 6 | 94 | [credit-top5 solution evaluation](06-credit-top5-solution-evaluation/credit-top5-solution-evaluation-all.ipynb) |
| 7 | 57 | [EDA, XGBoost, LightGBM & SHAP](07-eda-xgboost-lightgbm-shap/give-me-some-credit-eda-xgboost-lightgbm-shap.ipynb) |
| 8 | 53 | [Financial Distress Prediction](08-financial-distress-prediction/financial-distress-prediction.ipynb) |
| 9 | 51 | [MLJAR AutoML](09-mljar-automl/mljar-automl-givemesomecredit.ipynb) |
| 10 | 46 | [Starter: Give Me Some Credit](10-starter-give-me-some-credit/starter-give-me-some-credit.ipynb) |

Every directory also contains Kaggle's `kernel-metadata.json`.

## Validation

- 9 notebooks pass current `nbformat` validation unchanged.
- Rank 7 is readable as a 120-cell notebook but has a legacy Kaggle schema
  issue: a markdown cell contains an unexpected `execution_count` property.
  The downloaded original is deliberately preserved rather than rewritten.

## Refresh

```bash
./scripts/download_top_voted_givemesomecredit.sh
```

Vote totals change over time. The rankings and vote counts for this snapshot
are recorded in [`manifest.json`](manifest.json).

# Home Credit Model Stability benchmark

## Metric contract

- AUC, Brier, and KS use the persisted held-out test split.
- Lower Brier is better; higher AUC, KS, and Stability are better.
- Active features count non-zero persisted global-importance entries; ensembles and models without a native persisted importance are N/A.
- Stability: the week-based metric `mean(gini) + 88 * min(0, slope) - 0.5 * residual_std` over 19 out-of-time test weeks.
- Monotonic violations are reported only for models whose monotonicity is enforced and auditable; other models are N/A.
- Explanation time is median estimator-native local-attribution latency in milliseconds per row on 100 model-ready test rows, after one warm-up and across five measured runs; unsupported models and ensembles are N/A.
- EBM remains a candidate row and is not represented as a measured model.
- Gini is omitted because it is exactly `2 * AUC - 1` in these pipelines.

| Model | AUC | Brier | KS | Active features | Stability | Monotonic violations | Explanation time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM + SHAP | 0.830983 | 0.020465 | 0.508261 | 172 | 0.632246 | N/A | 2.277964 |
| Boosting ensemble | 0.830825 | 0.020362 | 0.512455 | N/A | 0.630952 | N/A | N/A |
| LightGBM + XGBoost + CatBoost | 0.830632 | 0.020354 | 0.512998 | N/A | 0.630392 | N/A | N/A |
| LightGBM + CatBoost | 0.830582 | 0.020342 | 0.512503 | N/A | 0.630473 | N/A | N/A |
| HistGradientBoosting | 0.829695 | 0.020441 | 0.512187 | N/A | 0.628804 | N/A | N/A |
| XGBoost | 0.829087 | 0.020434 | 0.511671 | 80 | 0.626988 | N/A | 0.379502 |
| CatBoost | 0.826387 | 0.020329 | 0.503434 | 79 | 0.620985 | N/A | 12.012315 |
| All-tree ensemble | 0.824513 | 0.042043 | 0.500359 | N/A | 0.619604 | N/A | N/A |
| Random Forest | 0.815398 | 0.169593 | 0.487949 | 80 | 0.597077 | N/A | N/A |
| LightGBM + CatBoost + Extra Trees | 0.806599 | 0.046275 | 0.484218 | N/A | 0.589728 | N/A | N/A |
| Monotonic LightGBM | 0.789966 | 0.020734 | 0.449310 | 7 | 0.544192 | 0 | 0.241708 |
| WoE scorecard | 0.783872 | 0.020888 | 0.445760 | 7 | 0.529584 | 0 | 0.002747 |
| Extra Trees | 0.726580 | 0.232020 | 0.346843 | 80 | 0.307400 | N/A | N/A |
| GAM | 0.707505 | 0.021247 | 0.320953 | 7 | 0.329556 | N/A | 0.000065 |
| Logistic | 0.667722 | 0.021540 | 0.240345 | 244 | 0.149123 | N/A | 0.000289 |
| EBM (not implemented) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

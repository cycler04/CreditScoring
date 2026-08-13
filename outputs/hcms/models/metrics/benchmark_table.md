# Home Credit Model Stability benchmark

## Metric contract

- AUC and KS use persisted held-out test metrics.
- Brier uses persisted held-out test predictions; it is N/A when an external run did not export those probabilities.
- Lower Brier is better; higher AUC, KS, and Stability are better.
- Active features count non-zero persisted global-importance entries; ensembles and models without a native persisted importance are N/A.
- Stability: the week-based metric `mean(gini) + 88 * min(0, slope) - 0.5 * residual_std` over 19 out-of-time test weeks.
- Monotonic violations are reported only for models whose monotonicity is enforced and auditable; other models are N/A.
- Explanation time is median estimator-native local-attribution latency in milliseconds per row on 100 model-ready test rows, after one warm-up and across five measured runs; unsupported models and ensembles are N/A.
- EBM remains a candidate row and is not represented as a measured model.
- Gini is omitted because it is exactly `2 * AUC - 1` in these pipelines.

| Model | AUC | Brier | KS | Active features | Stability | Monotonic violations | Explanation time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM + SHAP | 0.830990 | 0.020464 | 0.508316 | 171 | 0.632235 | N/A | 2.151808 |
| Boosting ensemble | 0.830895 | 0.020398 | 0.511410 | N/A | 0.631343 | N/A | N/A |
| LightGBM + XGBoost + CatBoost | 0.830579 | 0.020401 | 0.511543 | N/A | 0.630623 | N/A | N/A |
| LightGBM + CatBoost | 0.830391 | 0.020387 | 0.511740 | N/A | 0.630360 | N/A | N/A |
| HistGradientBoosting | 0.829881 | 0.020450 | 0.509605 | N/A | 0.629417 | N/A | N/A |
| XGBoost | 0.829274 | 0.020487 | 0.509198 | 156 | 0.627646 | N/A | 0.245751 |
| CatBoost | 0.826199 | 0.020401 | 0.503209 | 179 | 0.621252 | N/A | 12.862400 |
| All-tree ensemble | 0.813377 | 0.043158 | 0.478653 | N/A | 0.599673 | N/A | N/A |
| Random Forest | 0.804711 | 0.179692 | 0.458930 | 216 | 0.578134 | N/A | N/A |
| Monotonic LightGBM | 0.789966 | 0.020734 | 0.449310 | 7 | 0.544192 | 0 | 0.226033 |
| LightGBM + CatBoost + Extra Trees | 0.787630 | 0.046939 | 0.471654 | N/A | 0.552685 | N/A | N/A |
| WoE scorecard | 0.783872 | 0.020888 | 0.445760 | 7 | 0.529584 | 0 | 0.003239 |
| GAM | 0.707505 | 0.021247 | 0.320953 | 7 | 0.329556 | N/A | 0.000072 |
| Extra Trees | 0.688772 | 0.236052 | 0.305198 | 226 | 0.349000 | N/A | N/A |
| Logistic | 0.667722 | 0.021540 | 0.240345 | 244 | 0.149123 | N/A | 0.000171 |
| EBM (not implemented) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

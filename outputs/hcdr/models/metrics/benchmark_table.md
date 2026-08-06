# Home Credit Default Risk benchmark

## Metric contract

- AUC and KS use persisted held-out test metrics.
- Brier uses persisted held-out test predictions; it is N/A when an external run did not export those probabilities.
- Lower Brier is better; higher AUC, KS, and Stability are better.
- Active features count non-zero persisted global-importance entries; ensembles and models without a native persisted importance are N/A.
- Stability: N/A because HCDR has no time column and uses a stratified random split; the valid-test gap is not treated as temporal stability.
- Monotonic violations are reported only for models whose monotonicity is enforced and auditable; other models are N/A.
- Explanation time is median estimator-native local-attribution latency in milliseconds per row on 100 model-ready test rows, after one warm-up and across five measured runs; unsupported models and ensembles are N/A.
- EBM remains a candidate row and is not represented as a measured model.
- Gini is omitted because it is exactly `2 * AUC - 1` in these pipelines.

| Model | AUC | Brier | KS | Active features | Stability | Monotonic violations | Explanation time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM + XGBoost + CatBoost | 0.784172 | 0.066154 | 0.426841 | N/A | N/A | N/A | N/A |
| Boosting ensemble | 0.783686 | 0.066199 | 0.426248 | N/A | N/A | N/A | N/A |
| LightGBM + CatBoost | 0.783632 | 0.066182 | 0.427257 | N/A | N/A | N/A | N/A |
| CatBoost | 0.782643 | 0.066249 | 0.427480 | 164 | N/A | N/A | 21.742427 |
| XGBoost | 0.782425 | 0.066282 | 0.425585 | 298 | N/A | N/A | 0.889701 |
| LightGBM + SHAP | 0.780945 | 0.066379 | 0.424585 | 277 | N/A | N/A | 2.830395 |
| HistGradientBoosting | 0.778933 | 0.066565 | 0.419329 | N/A | N/A | N/A | N/A |
| All-tree ensemble | 0.775995 | 0.077788 | 0.412079 | N/A | N/A | N/A | N/A |
| LightGBM + CatBoost + Extra Trees | 0.773361 | 0.080438 | 0.411017 | N/A | N/A | N/A | N/A |
| FT-Transformer | 0.768974 | N/A | 0.403142 | N/A | N/A | N/A | N/A |
| Logistic | 0.765819 | 0.067466 | 0.398683 | 362 | N/A | N/A | 0.000350 |
| Random Forest | 0.755405 | 0.152800 | 0.382151 | 333 | N/A | N/A | N/A |
| Monotonic LightGBM | 0.747242 | 0.068602 | 0.368969 | 20 | N/A | 0 | 0.646866 |
| WoE scorecard | 0.745614 | 0.068718 | 0.366721 | 21 | N/A | 0 | 0.002833 |
| GAM | 0.740334 | 0.068824 | 0.356727 | 21 | N/A | N/A | 0.000130 |
| Extra Trees | 0.738789 | 0.195968 | 0.355240 | 343 | N/A | N/A | N/A |
| EBM (not implemented) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |


# Vì sao các model HCDR có kết quả gần nhau?

> **Câu hỏi:** các kiến trúc model trong benchmark Home Credit Default Risk (HCDR)
> tạo output như thế nào, và AUC gần nhau là do dữ liệu hay do kiến trúc?
>
> **Phạm vi:** Stage C của pipeline local, artifact đã lưu ngày 12/08/2026. Báo cáo
> phân tích benchmark offline, không suy diễn chất lượng ra quyết định tín dụng thực tế.
> Báo cáo này bổ sung cho [phân tích solution hạng 1](top1-model-usage-and-training.md),
> vốn nghiên cứu solution Kaggle lịch sử; nó không mô tả solution đó.

## Trả lời ngắn

**Các gradient-boosted tree gần nhau chủ yếu vì chúng nhìn thấy gần như cùng một bài
toán tabular và cùng các tín hiệu mạnh, không phải vì chúng là cùng một kiến trúc.**
LightGBM, XGBoost, CatBoost và HistGradientBoosting cùng học split theo ngưỡng và
tương tác thấp-bậc từ Stage C; ba model đầu xếp hạng gần như cùng các hồ sơ:
Spearman của output competition là `0.960010`–`0.978011`. AUC test của từng model
chỉ chênh `0.001698` từ LightGBM (`0.780945`) đến CatBoost (`0.782643`).

Tuy nhiên, **không đúng khi nói mọi model có hiệu năng giống nhau.** Boosted trees và
blend của chúng là một tầng riêng (`0.778933`–`0.784172` AUC); Logistic raw và
FT-Transformer thấp hơn, còn scorecard/GAM/Extra Trees thấp hơn nữa. Ensemble
ba-booster tăng `0.001529` AUC so với CatBoost, nên các output rất tương quan nhưng
vẫn còn một phần lỗi khác nhau để average khai thác.

Kết luận nguyên nhân theo mức bằng chứng:

1. **Verified:** cùng 175 feature Stage C, cùng membership train/valid/test cố định,
   cùng target nhị phân và cùng ROC-AUC khiến phần lớn điều kiện benchmark được chia sẻ.
2. **Verified:** các model mạnh đều đặt `EXT_SOURCE_1/2/3`, tuổi và các tỷ lệ tài
   chính/lịch sử tín dụng vào nhóm feature quan trọng; cách tạo `EXT_SOURCE_*` ở
   nguồn dữ liệu là không được công bố.
3. **Inferred:** với feature phần lớn là số đã aggregate, không có chuỗi thời gian,
   ảnh hay văn bản, nhiều boosted-tree implementation có thể tiến đến gần cùng thứ
   tự rủi ro. CatBoost có native categorical nhưng chỉ có 16 categorical feature;
   lợi thế kiến trúc này không đủ lớn để tách xa trên protocol hiện tại.
4. **Unknown:** chưa có repeated CV, paired significance test, ablation bỏ
   `EXT_SOURCE_*`, hay benchmark nhiều seed. Vì vậy không thể khẳng định các chênh
   AUC nhỏ là một statistical tie hoặc quy toàn bộ nguyên nhân cho một feature.

## Why — điều cần phân biệt

Nhìn một bảng AUC dễ dẫn đến hai kết luận sai trái:

- “model khác kiến trúc nhưng AUC gần nhau, vậy kiến trúc không quan trọng”; hoặc
- “model cao hơn `0.001` chắc chắn tốt hơn trong mọi population”.

Hai kết luận đều bỏ qua hai câu hỏi khác nhau: **model có xếp hạng cùng người vay
không**, và **benchmark có cho mỗi kiến trúc một cơ hội công bằng không**. Báo cáo
đo cả metric lẫn mức đồng thuận output, sau đó đối chiếu feature representation và
ràng buộc kiến trúc. ROC-AUC chỉ đo khả năng xếp hạng bad cao hơn good trên một test
split; nó không đo calibration, temporal stability, fairness hay hiệu quả kinh doanh.

## Protocol chung — nền tảng của phép so sánh

| Thành phần                    | Trạng thái đã xác minh                                                                                                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Population Stage C              | 307.511 dòng có nhãn, 48.744 dòng competition test; feature matrix có 175 predictor.                                                                                     |
| Split đánh giá               | Stratified random 60/20/20, seed 42: 184.506 train, 61.502 validation, 61.503 test;**không phải** out-of-time validation.                                             |
| Target test                     | 4.965 bad trên 61.503 dòng, bad rate`0.080728`.                                                                                                                           |
| Input cho LR/tree trừ CatBoost | 175 raw feature đi qua median imputation + missing indicator + scaling cho numeric, constant imputation + one-hot (`min_frequency=0.01`) cho categorical; thành 362 cột. |
| Input CatBoost                  | 175 raw feature, trong đó 16 categorical được giữ native; missing numeric do CatBoost xử lý.                                                                          |
| Ensemble                        | Trung bình xác suất với trọng số bằng nhau, thành viên cố định; không tune trọng số trên test.                                                                |

Nguồn: [pipeline HCDR](../../../../src/home_credit_default_rate/pipeline.py),
[run summary Stage C](../../../../outputs/hcdr/run_summary.json),
[split membership](../../../../datasets/processed/hcdr/split_membership.csv) và
[benchmark protocol](../../../../outputs/hcdr/models/metrics/benchmark_protocol.json).

Điều này làm benchmark công bằng ở cấp **dòng/split**, nhưng không hoàn toàn đồng
nhất ở cấp **representation**: CatBoost nhận category native; GAM, monotonic
LightGBM và WoE scorecard bị giới hạn 21 feature đã chọn; FT-Transformer có pipeline
token hóa riêng. Vì thế, bảng dưới là so sánh hữu ích về output thực tế nhưng không
phải ablation thuần túy của một thuật toán.

## How — kiến trúc nào đang được so sánh?

| Nhóm / model        | Kiến trúc và input thực tế                                                                                     |           AUC test | Ý nghĩa của output                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- | -----------------: | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic raw         | Logistic Regression tuyến tính trên 362 cột đã transform.                                                     |           0.765819 | Baseline tuyến tính: không tự học interaction.                                                                                      |
| WoE scorecard        | Binning + WoE, rồi Logistic Regression trên 21 feature.                                                           |           0.745614 | Điểm rủi ro có hướng/biến đổi auditable, đánh đổi capacity.                                                                 |
| GAM                  | Spline bậc 3, 4 knots cho 21 feature rồi Logistic Regression.                                                     |           0.740334 | Phi tuyến từng biến, vẫn không học interaction tổng quát.                                                                        |
| Monotonic LightGBM   | GBDT 24 leaves trên 21 WoE feature, ép 21 monotone constraint.                                                    |           0.747242 | Cho phép phi tuyến có kiểm soát nhưng hạn chế không gian hàm.                                                                  |
| LightGBM             | Leaf-wise GBDT, 32 leaves, 1.000 trees tối đa, learning rate 0,02.                                                |           0.780945 | Học threshold và interaction từ 362 cột sparse.                                                                                      |
| XGBoost              | Histogram gradient boosting, max depth 6, 1.000 trees tối đa, learning rate 0,02.                                 |           0.782425 | Cùng họ boosted tree, control capacity bằng depth.                                                                                    |
| CatBoost             | Symmetric trees depth 7, 1.000 trees đã fit, native categorical/CTR,`Plain` boosting.                           |           0.782643 | Khai thác category không one-hot và interaction cân xứng.                                                                           |
| HistGradientBoosting | Histogram GBDT, tối đa 31 leaves, 300 iterations, dense 362 cột.                                                 |           0.778933 | Một implementation boosted tree khác, vẫn cùng inductive bias gần booster.                                                          |
| Random Forest        | 300 cây bagging, depth 14,`min_samples_leaf=20`, feature subsampling.                                            |           0.755405 | Average cây độc lập, không tuần tự sửa residual như boosting.                                                                   |
| Extra Trees          | 300 randomized trees, depth 14,`min_samples_leaf=20`.                                                             |           0.738789 | Random split mạnh hơn; bias tăng trên tín hiệu threshold tinh tế.                                                                 |
| FT-Transformer       | 159 numeric token + 16 categorical embedding, token dim 64, 3 encoder layers, 8 heads, CLS head; 131.393 parameter. |           0.768974 | Attention học interaction toàn cục, nhưng run hiện có budget 15 epoch, hoàn tất 13 epoch và không lưu held-out probabilities. |
| Equal-weight blends  | Average xác suất của 2–6 tree model, không có stacker/weight learned.                                         | 0.773361–0.784172 | Giảm một phần variance/lỗi riêng nếu thành viên có residual diversity.                                                          |

Các cấu hình ở bảng là **Verified** từ artifact `joblib` đã fit và
[cấu hình FT-Transformer](../../../../outputs/hcdr/kaggle_ft_transformer/experiment_config.json),
không suy từ tên model. Hình thức huấn luyện của GAM/monotonic challenger nằm trong
[script benchmark interpretable](../../../../scripts/pipelines/train_interpretable_home_credit_benchmarks.py).

## Output thực tế: model nào thực sự gần nhau?

### 1. Có ba tầng AUC, không phải một cụm duy nhất

| Tầng                                        | Model (AUC test)                                                                                                             | Diễn giải đúng                                                               |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Boosted tree / blend                         | CatBoost 0.782643, XGBoost 0.782425, LightGBM 0.780945, HistGradientBoosting 0.778933; blend 2–4 booster 0.783632–0.784172 | Khá gần nhau trên cùng feature matrix.                                       |
| Baseline giàu capacity vừa phải           | FT-Transformer 0.768974, Logistic raw 0.765819, Random Forest 0.755405                                                       | Không đạt bằng booster; FT không phải bằng chứng Transformer luôn kém. |
| Bị giới hạn representation/regularization | Monotonic LGBM 0.747242, WoE 0.745614, GAM 0.740334, Extra Trees 0.738789                                                    | Capacity, monotonicity hoặc random split tree tạo trade-off rõ rệt.          |

Nguồn đầy đủ (bao gồm KS và Brier) là
[benchmark table](../../../../outputs/hcdr/models/metrics/benchmark_table.md) và
[metrics CSV](../../../../outputs/hcdr/models/metrics/metrics.csv). Đây là metric
persisted trên cùng held-out test; FT-Transformer được chạy trên cùng membership nhưng
chỉ có AUC/Gini/KS vì run Kaggle không export test probability local.

![Dashboard benchmark HCDR: AUC, Gini và KS trên held-out test](../../../../outputs/hcdr/models/metrics/benchmark_dashboard.png)

*Hình 1 — Dashboard cho thấy nhóm boosted tree/blend gần nhau ở cả ba metric, còn
những architecture/representation bị giới hạn tạo khoảng cách rõ. Mở artifact:
[PNG dashboard](../../../../outputs/hcdr/models/metrics/benchmark_dashboard.png).*

### 2. Ba booster chính không chỉ gần AUC mà còn gần thứ tự rủi ro

Các con số sau được tính lại từ model artifact và Stage C held-out test, đồng thời
được kiểm tra không lệch AUC đã persisted. `MAD` là trung bình `|p_a - p_b|`; Spearman
so sánh thứ tự score nên trực tiếp liên quan ROC-AUC.

| Cặp trên 61.503 test row | Pearson probability | Spearman rank | MAD probability |
| -------------------------- | ------------------: | ------------: | --------------: |
| LightGBM — XGBoost        |            0.980451 |      0.981417 |        0.009974 |
| LightGBM — CatBoost       |            0.962861 |      0.964999 |        0.013549 |
| XGBoost — CatBoost        |            0.964132 |      0.963681 |        0.013753 |

Từ output competition 48.744 dòng không nhãn, cùng kết luận vẫn giữ: Spearman lần lượt
là `0.978011`, `0.961124`, `0.960010`. Như vậy đây không chỉ là AUC tình cờ gần nhau:
ba model đang xếp hạng phần lớn hồ sơ theo cùng hướng. Các submission đều có ID duy
nhất và xác suất hữu hạn `[0, 1]`.

Ensemble vẫn có lợi ích nhỏ: LightGBM + XGBoost + CatBoost đạt `0.784172`, cao hơn
CatBoost `0.001529`, XGBoost `0.001746` và LightGBM `0.003227`. Tương quan cao không
phải tương quan bằng một; phần disagreement còn lại là điều average khai thác.

![ROC curves của các model và ensemble đã tái dựng từ held-out test](../../../../outputs/hcdr/models/metrics/roc_auc_curve.png)

*Hình 2 — Các đường ROC của booster và blend chồng sát nhau, trực quan hóa nhận định
“khác kiến trúc nhưng ranking gần nhau”. Hình chỉ gồm model có held-out prediction
local; FT-Transformer, GAM và monotonic LightGBM không nằm trong curve này.
Mở artifact: [PNG ROC](../../../../outputs/hcdr/models/metrics/roc_auc_curve.png).*

### 3. “Khác output” không mặc nhiên là “nhiều thông tin hơn”

Extra Trees và Random Forest có Brier lần lượt `0.195968` và `0.152800`, cao hơn rõ
so với `0.066154` của blend ba booster. Khi đưa Extra Trees vào blend LightGBM +
CatBoost, AUC giảm xuống `0.773361`; average tất cả cây chỉ `0.775995`. Đây là
**Verified evidence** rằng diversity không được calibration/quality phù hợp có thể
pha loãng ranking tốt, thay vì tự động nâng hiệu năng.

FT-Transformer có Spearman output competition `0.919000` với LightGBM (vẫn cùng
hướng xếp hạng) nhưng Pearson chỉ `0.799062` và MAD `0.339170`. Khác biệt scale này
phù hợp với việc run dùng `BCEWithLogitsLoss(pos_weight=non_event/event)`; không có
Brier test để kết luận calibration. Do đó không được so sánh trị tuyệt đối xác suất
FT với tree như một thước đo chất lượng.

## Vì sao dữ liệu làm booster hội tụ?

### Tín hiệu mạnh được nhiều model cùng tìm thấy

Top importance không cùng đơn vị giữa các algorithm, nên không cộng/trừ trực tiếp.
Nhưng thứ tự feature có sự giao nhau rõ:

| Model                   | Một số feature đứng đầu đã lưu                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| LightGBM                | `EXT_SOURCE_2`, `EXT_SOURCE_1`, `DAYS_BIRTH`, `EXT_SOURCE_3`, `AMT_ANNUITY`                  |
| XGBoost                 | missing indicator của credit-bureau enquiry,`EXT_SOURCE_3`, `EXT_SOURCE_2`, missingness của debt |
| CatBoost                | `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `DAYS_BIRTH`, `CREDIT_GOODS_RATIO`           |
| Random/Extra Trees      | `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, tuổi/credit-bureau/lịch sử repayment          |
| WoE/GAM/monotonic model | ba`EXT_SOURCE_*`, employment/age ratio, bureau, previous application và card utilization            |

Nguồn: các CSV trong
[feature importance](../../../../outputs/hcdr/models/feature_importance/). Ba
`EXT_SOURCE_*` là normalized external scores theo dictionary HCDR; cách upstream tạo
chúng là **Unknown**, vì vậy không gọi chúng là FICO, PD đã calibration hoặc một
nguồn bureau cụ thể.

Dataset cũng có missingness giàu thông tin: `EXT_SOURCE_1` thiếu 56,39%, nhiều feature
building/credit-card/bureau-balance thiếu xấp xỉ 50–72%. Pipeline one-hot/numeric thêm
missing indicator, còn CatBoost xử lý missing/category native. **Inferred:** khi
missingness và vài continuous risk signal đã tách tốt good/bad, các booster khác nhau
có thể học những boundary tương tự, nên lợi thế xử lý category native không bùng nổ.
Xem [column profile](../../../../outputs/hcdr/eda/column_profile.csv) để tái kiểm số
thiếu và [feature flow](../feature-extraction/src-data-extraction-and-flow-report-vi.md)
để biết cách aggregate về một dòng `SK_ID_CURR`.

### Feature engineering tác động lớn hơn đổi booster trong run này

Khi đi từ Stage A (127 feature) sang B (145) rồi C (175), AUC test tăng cho cả ba
family chung representation:

| Model         |  Stage A |  Stage B |  Stage C | Tăng A → C |
| ------------- | -------: | -------: | -------: | -----------: |
| Logistic raw  | 0.751937 | 0.755380 | 0.765819 |    +0.013882 |
| LightGBM      | 0.764614 | 0.771951 | 0.780945 |    +0.016331 |
| XGBoost       | 0.766032 | 0.772842 | 0.782425 |    +0.016393 |
| WoE scorecard | 0.733826 | 0.736836 | 0.745614 |    +0.011788 |

Đây là **Verified association**, không phải causal ablation hoàn hảo: Stage thay đổi
feature set và có thể cả data-preparation consequences. Dù vậy, việc mọi family cùng
tăng khoảng `0.012`–`0.016` củng cố giả thuyết rằng information representation là
driver lớn trong benchmark này.

## Kiến trúc vẫn tạo khác biệt ở đâu?

| Quan sát                                  | Cơ chế hợp lý                                                                                                                       | Trạng thái                                                                                           |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Boosting hơn Random/Extra Trees           | Cây được thêm tuần tự để sửa residual; bagging/randomized split không làm vậy.                                             | **Inferred**, phù hợp thứ tự AUC đã đo.                                                   |
| CatBoost/XGBoost nhỉnh LightGBM rất nhỏ | Khác cách grow tree, histogram/regularization và categorical handling tạo residual diversity nhỏ.                                  | **Inferred**; không có ablation từng mechanism.                                               |
| Model 21 feature thấp hơn                | Binning/WoE, additive spline và monotonic constraints bỏ bớt interaction/capacity để đổi lấy auditability.                      | **Verified** về constraint và metric; mức đóng góp từng constraint là **Unknown**. |
| FT-Transformer chưa vượt tree           | Model chỉ 131k parameter, budget 15 epoch nhưng hoàn tất 13 với early stop, còn benchmark là tabular static và class imbalance. | **Inferred**; không thể kết luận Transformer thua sau một config/run.                       |
| Blend ba booster tốt nhất                | Xác suất không hoàn toàn giống nhau, nên equal average giảm một phần lỗi.                                                    | **Verified** về correlation/AUC; lý giải variance là **Inferred**.                     |

## Giới hạn và việc cần kiểm chứng tiếp

- **Không có time column:** random split không kiểm tra drift hay OOT generalization.
- **Không có repeated CV/seed:** không gắn nhãn “statistically tied” cho chênh AUC
  nhỏ. Bước kiểm chứng phù hợp là 5-fold OOF, paired bootstrap/DeLong trên OOF và CI
  của chênh AUC, giữ mọi preprocessing train-fold-only.
- **Không có ablation nguồn tín hiệu:** cần rerun bỏ lần lượt `EXT_SOURCE_*`, missing
  indicators, auxiliary aggregates và categorical features trên cùng folds để phân
  rã nguyên nhân dataset vs architecture.
- **Calibration chưa đồng đều:** Brier FT là `N/A`; trước threshold/approval policy
  cần export held-out probabilities và kiểm calibration (reliability curve, Brier,
  calibration slope) trên validation độc lập.
- **Ensemble hiện là equal weight:** chỉ thử constrained weight/stacking trên OOF,
  không chọn trọng số theo held-out test.

## Tái lập từ artifact hiện có

Không cần retrain để dựng lại metric, table và chart:

```bash
uv run python scripts/pipelines/generate_home_credit_benchmark_tables.py
uv run python scripts/pipelines/generate_hc_diagrams.py --dataset hcdr --with-predictions
```

Lệnh đầu tái dựng metric từ held-out matrix và model artifact, đồng thời kiểm AUC/KS
không lệch quá tolerance `1e-5`; lệnh sau sinh lại dashboard/curves. FT artifact có
thể kiểm schema/provenance riêng bằng:

```bash
uv run python scripts/pipelines/validate_hcdr_ft_transformer_artifacts.py
```

## Nguồn local

- [Bảng benchmark HCDR](../../../../outputs/hcdr/models/metrics/benchmark_table.md)
  và [dashboard](../../../../outputs/hcdr/models/metrics/benchmark_dashboard.png).
- [Stage C metrics](../../../../outputs/hcdr/models/metrics_C.csv),
  [run summary](../../../../outputs/hcdr/run_summary.json),
  [feature matrix](../../../../datasets/processed/hcdr/feature_matrix.parquet).
- [Pipeline HCDR](../../../../src/home_credit_default_rate/pipeline.py),
  [benchmark-table renderer](../../../../scripts/pipelines/generate_home_credit_benchmark_tables.py),
  [FT notebook builder](../../../../scripts/pipelines/build_hcdr_ft_transformer_notebook.py).
- [FT experiment config](../../../../outputs/hcdr/kaggle_ft_transformer/experiment_config.json)
  và [training history](../../../../outputs/hcdr/kaggle_ft_transformer/training_history.csv).

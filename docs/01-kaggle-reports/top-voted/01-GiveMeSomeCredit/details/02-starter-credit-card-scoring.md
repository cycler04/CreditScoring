# #2 — Starter: credit card scoring

## Hồ sơ nguồn

- Tác giả: `riteshrhyme`.
- Kaggle: [Starter: credit card scoring](https://www.kaggle.com/code/riteshrhyme/starter-credit-card-scoring-bbe98584-0).
- Snapshot: hạng 2, **233 vote**, ngày 2026-07-28.
- File đã đọc:
  [`starter-credit-card-scoring-bbe98584-0.ipynb`](../../../../notebooks/top-voted/GiveMeSomeCredit/02-starter-credit-card-scoring/starter-credit-card-scoring-bbe98584-0.ipynb).
- Quy mô: 157 cell (152 code, 5 markdown), 53,161 byte.
- SHA-256:
  `a898c35e2a08251874ab8931d5de21bffd34c0917096da2d7d36e20de3115d7e`.
- Trạng thái artifact: không có output và không có cell đã execute.

## Tóm tắt

Notebook thử phạm vi rất rộng:

- EDA và outlier;
- xóa missing/zero income;
- so sánh 10 classifier;
- StandardScaler/MinMaxScaler;
- class weight;
- feature importance;
- random/grid search;
- voting ensemble;
- SuperLearner.

Độ rộng này hữu ích như checklist học tập, nhưng snapshot không phải một chương
trình chạy được. Nhiều lỗi biến, API và validation khiến mọi con số viết tay
không có provenance đáng tin. Đây là notebook yếu nhất trong top 3 về
reproducibility.

## Khả năng chạy lại

**Verified — không chạy tuần tự từ đầu đến cuối.** Một số lỗi xuất hiện trước khi
đến phần modeling:

- cell 13 gọi `test_df.head(20)` nhưng không có cell nào tạo `test_df`;
- cell split gọi `train_test_split()` không truyền đối số rồi gán bốn biến;
- sau đó code bỏ qua split và đặt `X_train`, `Y_train` bằng toàn bộ dataframe;
- phần tuning dùng cả `Y_train` và `y_train`, nhưng `y_train` không được định nghĩa;
- `GridSearch` được gọi với đối số `scoring` dù constructor không nhận đối số đó;
- cuối notebook dùng `Y_test` dù không có holdout hợp lệ;
- Logistic Regression với `penalty='l1'` không chỉ định solver, phụ thuộc version
  và có thể lỗi;
- cài `mlens` giữa notebook nhưng không pin version.

Do không có output, không thể biết tác giả đã chạy theo thứ tự khác, dùng state
còn sót từ session cũ, hay chỉnh tay trước khi publish.

## Dữ liệu và làm sạch

Notebook đọc `cs-training.csv`, bỏ cột index, rồi:

1. xóa mọi dòng `MonthlyIncome` missing;
2. xóa mọi dòng `MonthlyIncome == 0`;
3. phát hiện outlier bằng IQR;
4. xóa thêm nhiều dòng **chỉ khi target bằng 0**, ví dụ:
   - delinquency > 50 và non-default;
   - `DebtRatio > 5` và non-default;
   - income > 30,000 và non-default;
   - open credit lines > 20 và non-default;
   - dependents > 6 và non-default.

**Verified — target leakage/selection bias nghiêm trọng.** Dùng
`SeriousDlqin2yrs` để quyết định giữ hay xóa dòng làm thay đổi có chủ đích phân
phối lớp theo feature. Quy tắc này không thể áp dụng khi scoring hồ sơ mới vì lúc
đó target chưa tồn tại. Metric sau bước làm sạch không còn ước lượng bài toán
competition ban đầu.

Ngoài ra, xóa toàn bộ missing income làm mất gần một phần năm dữ liệu và không
kiểm tra missingness có mang tín hiệu rủi ro hay không.

## So sánh model ban đầu

Hàm `BasedLine2` dùng 10-fold `StratifiedKFold` và so sánh:

- Logistic Regression, LDA, KNN;
- Decision Tree, Naive Bayes, SVM;
- AdaBoost, Gradient Boosting;
- Random Forest, Extra Trees.

Nó chạy weighted F1 và accuracy, sau đó thử scaling và class weights. Scaler được
đặt trong `Pipeline`, là một điểm đúng: thống kê scale được fit riêng theo từng
fold. scikit-learn cũng khuyến nghị pipeline để tránh leakage trong CV
([Pipeline](https://scikit-learn.org/stable/modules/compose.html)).

Tuy nhiên:

- metric chính của competition là ROC-AUC, không phải accuracy/weighted F1;
- dataframe đã bị xóa dòng theo target trước CV;
- `StratifiedKFold(n_splits=10, random_state=SEED)` không bật `shuffle`, nên
  `random_state` không có tác dụng trong các version hiện đại và có thể lỗi tùy
  version;
- không có output để xác nhận model nào thực sự chạy.

## Feature selection và tuning

Extra Trees được fit trên toàn bộ dữ liệu đã làm sạch để chọn sáu feature, rồi
CV được chạy trên chính dữ liệu đó. Đây là feature-selection leakage: fold test
đã góp phần quyết định feature. Hướng dẫn scikit-learn nêu rõ feature selection
phải nằm trong pipeline/fold train
([Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)).

Grid/random search mặc định tối ưu `estimator.score` (thường là accuracy) vì code
không truyền scoring vào `GridSearchCV`/`RandomizedSearchCV`. Sau khi chọn model,
notebook lại gọi:

```python
Prediction_LR = ...BestModelPridict(X_train)
classification_report(Y_train, Prediction_LR)
```

Tức là report trên chính dữ liệu fit. Hai số `0.929252` và `0.926401` được gõ tay
vào bảng `hpt`; chúng không được nối với object metric hay output đã lưu.

**Kết luận:** các số này không phải bằng chứng generalization và không nên trích
làm benchmark.

## Ensemble

Notebook dựng VotingClassifier từ chín model và thử SuperLearner. Ý tưởng error
correlation giữa base learners là hợp lý, nhưng implementation không hoàn tất:

- `X_test` dựa trên `test_df` không tồn tại và còn kỳ vọng cột target trong test;
- `train_predict` cấp phát theo `ytest.shape` dù test thực của competition không
  có nhãn;
- `Y_test` không được định nghĩa;
- không có ROC-AUC holdout hoặc submission file;
- stacking không được đánh giá out-of-fold bằng artifact.

Ensemble vì thế là skeleton, không phải kết quả thực nghiệm.

## Phần có thể tái sử dụng

- Danh sách model baseline để thiết kế benchmark có kiểm soát.
- Đặt scaler và estimator trong `Pipeline`.
- So sánh class-weighted và unweighted models.
- Kiểm tra error correlation trước khi ensemble.

## Phần phải loại bỏ hoặc viết lại

1. Không xóa/cap dòng dựa trên target.
2. Tạo split trước EDA có tính quyết định, imputation, outlier rules và feature
   selection.
3. Dùng ROC-AUC từ probability/decision score; thêm Gini/KS theo pipeline local.
4. Đặt feature selection trong pipeline hoặc nested CV.
5. Dùng inner CV cho tuning và outer CV/holdout để báo cáo; scikit-learn cảnh báo
   non-nested selection cho score lạc quan
   ([Nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)).
6. Pin version và seed; loại bỏ notebook state ẩn.
7. Xuất metric table từ object kết quả, không gõ tay.

## Đánh giá cuối

- **Verified:** notebook local không chạy tuần tự.
- **Verified:** làm sạch dùng target và đánh giá tuning trên train.
- **Verified:** metric chính không khớp competition.
- **Unknown:** mọi score/model ranking vì không có output hoặc artifact.

Kết luận: chỉ dùng notebook như inventory ý tưởng. Không dùng code, con số hoặc
quy tắc làm sạch của notebook làm nền cho pipeline local.

## Nguồn

- Ritesh Rhyme, [Starter: credit card scoring](https://www.kaggle.com/code/riteshrhyme/starter-credit-card-scoring-bbe98584-0),
  Kaggle, truy cập 2026-07-29.
- scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html),
  truy cập 2026-07-29.
- scikit-learn, [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html),
  truy cập 2026-07-29.
- scikit-learn, [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
  truy cập 2026-07-29.


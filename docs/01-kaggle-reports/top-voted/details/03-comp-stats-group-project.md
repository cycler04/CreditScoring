# #3 — Comp Stats Group Data Project

## Hồ sơ nguồn

- Tác giả: `simonpfish`.
- Kaggle: [Comp Stats Group Data Project — Final](https://www.kaggle.com/code/simonpfish/comp-stats-group-data-project-final).
- Snapshot: hạng 3, **205 vote**, ngày 2026-07-28.
- File đã đọc:
  [`comp-stats-group-data-project-final.ipynb`](../../../../notebooks/top-voted/GiveMeSomeCredit/03-comp-stats-group-project/comp-stats-group-data-project-final.ipynb).
- Quy mô: 51 cell (22 code, 29 markdown), 19,556 byte.
- SHA-256:
  `fa4ab36b395fe6b6d5bf40fb866ac7658b98eeaf8bf02392f2a9f93028bb3087`.
- Trạng thái artifact: không có output và không có cell đã execute.

## Tóm tắt

Đây là notebook rõ ràng nhất trong top 3 về lập luận EDA. Nhóm tác giả:

1. điều tra imbalance và bốn vùng anomaly;
2. tạo nhiều biến thể dữ liệu;
3. xây `Tester` để chạy ROC-AUC cross-validation cho nhiều dataset/model;
4. chọn Random Forest và tune `max_depth`, `n_estimators`;
5. thử KNN và thảo luận thất bại của neural network.

Notebook báo cáo Random Forest depth 9, 16 trees trên dataset bỏ utilization
outliers đạt AUC **0.8662**. Đây là claim có giá trị tham khảo, nhưng snapshot
không có output và quy trình dùng cùng CV cho model selection và reporting, nên
không phải test estimate độc lập.

## EDA và anomaly

### Class imbalance

Notebook tính mean của `SeriousDlqin2yrs` và nhận ra accuracy có thể bị đánh lừa.
Phần neural network mô tả model đạt khoảng 93% accuracy bằng cách đoán một lớp,
rồi chủ động không dùng kết quả đó. Đây là bài học đúng: metric phải phản ánh khả
năng xếp hạng/phân biệt, không chỉ tỷ lệ dự đoán đúng ở một threshold.

### `DebtRatio`

Nhóm phát hiện percentile 97.5 khoảng 3,489 và kiểm tra quan hệ với
`MonthlyIncome`. Họ quan sát phần lớn các dòng cực trị thiếu income, và nhiều dòng
có income 0/1 trùng target. Đây là EDA liên biến hữu ích, tốt hơn cap theo một
ngưỡng đơn biến mà không xét missingness.

Tuy nhiên việc gọi đây chắc chắn là “data-entry error” là **Inferred**, không được
chứng minh bằng data dictionary hay provenance gốc.

### Delinquency 96/98

Notebook phát hiện khoảng trống giữa 17 và 96, đồng thời ba biến delinquency cùng
nhận 96/98 trên các dòng giống nhau. Họ đề xuất thay giá trị >90 bằng 18 thay vì
xóa, nhằm giảm ảnh hưởng lên SVM.

Phát hiện pattern là **Verified từ code/EDA**; ý nghĩa chính xác của 96/98 vẫn
**Unknown** nếu không có xác nhận từ chủ dữ liệu. Trong pipeline nên tạo anomaly
flag, thử cả giữ nguyên/cap/missing, rồi đánh giá trên validation.

### Revolving utilization

Notebook chia các vùng 0.9–4, 4–10 và >10. Nhóm >10 nhỏ và có bad rate không cao
như kỳ vọng, nên bị loại trong một dataset thử nghiệm. Đây là giả thuyết thực
nghiệm hợp lý, nhưng threshold 10 được chọn sau khi nhìn target trên toàn bộ dữ
liệu; nó phải được fit/chọn chỉ trên train.

## Imputation và các biến thể dữ liệu

Notebook thử hồi quy tuyến tính để dự đoán `MonthlyIncome`, nhưng đo `R²` trên
chính dữ liệu fit và thấy thấp. Sau đó chọn median fill; dependents missing được
thay 0.

Các dataset so sánh:

- drop missing;
- median fill;
- median fill rồi bỏ `DebtRatio` > 3489.025;
- bỏ utilization > 10;
- thay ba mã delinquency > 90 bằng 18.

**Verified — preprocessing leakage.** Median và threshold/rule được tính hoặc chọn
trên toàn bộ dataframe trước khi truyền vào CV. scikit-learn khuyến nghị mọi
imputer và transform phải fit trong từng fold train, tốt nhất qua `Pipeline`
([Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)).

## `Tester` và validation

`Tester` gọi:

```python
cross_validate(clf, X, Y, scoring=['roc_auc'], cv=4, n_jobs=-1)
```

Điểm tốt:

- dùng ROC-AUC đúng với mục tiêu competition;
- cùng một harness cho mọi model/dataset;
- báo mean và standard deviation qua fold;
- Random Forest có `random_state=0`.

Giới hạn:

- mặc định sample 80,000 dòng bằng `DataFrame.sample()` không có
  `random_state`, nên kết quả không tái lập;
- cột `Unnamed: 0` không được drop trong `Tester`, có thể đi vào model như một
  feature ID vô nghĩa;
- preprocessing nằm ngoài CV;
- không lưu fold indices hay out-of-fold predictions;
- không có validation/test khóa riêng.

ROC-AUC được tính từ ranking score và phù hợp hơn accuracy cho bài toán này
([`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)).

## Tuning và claim AUC 0.8662

Notebook thêm 50 Random Forest:

- `max_depth` từ 5 đến 9;
- `n_estimators` từ 10 đến 19.

Sau đó `runTests()` xếp hạng tất cả kết quả và markdown công bố cấu hình tốt nhất.
Không có outer CV. Vì cùng các fold vừa dùng để chọn dataset, threshold xử lý
outlier và hyperparameter cũng được dùng để báo cáo score, 0.8662 có selection
bias. Tài liệu scikit-learn mô tả non-nested CV theo đúng dạng này là lạc quan;
nested CV tách vòng chọn tham số khỏi vòng ước lượng generalization
([Nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)).

Trạng thái metric:

- **Verified:** markdown của notebook ghi AUC 0.8662 và cấu hình depth 9,
  16 estimators.
- **Unknown:** giá trị chính xác có tái lập được từ snapshot hay không, vì output
  đã bị xóa và sample không seed.
- **Inferred:** score có khả năng lạc quan do model/dataset selection trên cùng CV.

## Vấn đề với kết luận pháp lý về `age`

Notebook nhìn phân phối default theo tuổi, dẫn một bài báo về mortgage rồi kết luận
“we're good to go, legally.” Kết luận đó không được hỗ trợ bởi một histogram.

Regulation B hiện xem age là prohibited basis, đồng thời có các điều kiện hẹp cho
việc dùng age trong hệ thống chấm điểm “empirically derived, demonstrably and
statistically sound”; người từ 62 tuổi trở lên không được nhận factor kém thuận
lợi hơn nhóm nonelderly được ưu ái nhất
([CFPB official interpretation §1002.6](https://www.consumerfinance.gov/rules-policy/regulations/1002/Interp-6),
[Regulation B definitions](https://www.consumerfinance.gov/rules-policy/regulations/1002/2/)).

Do đó:

- **Verified:** suy luận pháp lý của notebook là quá mức và không đủ bằng chứng.
- **Không suy rộng:** repo này là benchmark thực hành, không phải quyết định tín
  dụng production hay tư vấn pháp lý.

## Phần nên tái sử dụng

- Cách điều tra anomaly theo vùng và bad rate, không chỉ nhìn boxplot.
- Harness so sánh nhiều dataset/model bằng cùng metric.
- Thảo luận thẳng về failure của accuracy trên lớp mất cân bằng.
- Thử sensitivity của các cách xử lý 96/98, utilization và missing income.

## Phần phải viết lại

1. Drop cột ID ngay khi load.
2. Split stratified trước; fit median, anomaly rules và threshold chỉ trên train.
3. Đặt preprocessing + model trong pipeline của từng fold.
4. Seed cả sampling lẫn model; lưu fold indices và OOF predictions.
5. Dùng inner CV để chọn dataset/rule/hyperparameter, validation hoặc outer CV để
   so sánh, test khóa riêng để báo cáo cuối.
6. Không diễn giải anomaly là lỗi nhập liệu nếu chưa có data provenance.
7. Tách model-performance analysis khỏi legal/fair-lending review.

## Đánh giá cuối

- **Verified:** EDA anomaly là phần mạnh nhất trong top 3.
- **Verified:** ROC-AUC và harness so sánh model phù hợp hơn metric của notebook #2.
- **Inferred:** AUC 0.8662 bị optimistic selection bias.
- **Unknown:** khả năng tái lập chính xác vì không có output, sample không seed.
- **Verified:** claim “legally good to go” không có đủ cơ sở.

Kết luận: đây là notebook đáng đọc nhất để thiết kế EDA, nhưng validation cần được
xây lại hoàn toàn trước khi dùng bất kỳ metric nào.

## Nguồn

- Simon Fish và cộng sự, [Comp Stats Group Data Project — Final](https://www.kaggle.com/code/simonpfish/comp-stats-group-data-project-final),
  Kaggle, truy cập 2026-07-29.
- scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html),
  truy cập 2026-07-29.
- scikit-learn, [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
  truy cập 2026-07-29.
- CFPB, [Official interpretation of Regulation B §1002.6](https://www.consumerfinance.gov/rules-policy/regulations/1002/Interp-6),
  phiên bản hiện hành, truy cập 2026-07-29.
- CFPB, [Regulation B §1002.2 — Definitions](https://www.consumerfinance.gov/rules-policy/regulations/1002/2/),
  phiên bản hiện hành, truy cập 2026-07-29.


# Top-voted notebooks — GiveMeSomeCredit

## Câu hỏi và phạm vi

Báo cáo này trả lời: các notebook được vote nhiều nhất trong competition
GiveMeSomeCredit làm gì, bằng chứng nào đáng tin, lỗi phương pháp nào cần tránh, và
phần nào có thể tái sử dụng cho pipeline local?

- Ngày nghiên cứu: **2026-07-29**.
- Nguồn xếp hạng: snapshot Kaggle Code tab ngày **2026-07-28**, sort
  `voteCount` giảm dần, lưu tại
  [`manifest.json`](../../../notebooks/top-voted/GiveMeSomeCredit/manifest.json).
- Phạm vi đọc sâu: **top 3**. Cả ba file `.ipynb` đều được đọc trực tiếp; không
  chỉ dựa vào tiêu đề, mô tả hoặc search snippet.
- Đây là **community notebooks**, không phải paper học thuật và không phải code
  của ba đội đứng đầu leaderboard năm 2011.
- Không rerun notebook. Các snapshot đã xóa toàn bộ output và
  `execution_count`, vì vậy metric chỉ được xem là claim của tác giả nếu không
  thể tính lại từ artifact.

## Xếp hạng snapshot

| Hạng | Vote | Notebook                                                                                                                 | Mức kiểm tra |
| ----: | ---: | ------------------------------------------------------------------------------------------------------------------------ | -------------- |
|     1 |  336 | [Credit ScoreCard example](https://www.kaggle.com/code/orange90/credit-scorecard-example)                                 | Đọc sâu     |
|     2 |  233 | [Starter: credit card scoring](https://www.kaggle.com/code/riteshrhyme/starter-credit-card-scoring-bbe98584-0)            | Đọc sâu     |
|     3 |  205 | [Comp Stats Group Data Project](https://www.kaggle.com/code/simonpfish/comp-stats-group-data-project-final)               | Đọc sâu     |
|     4 |  199 | [Modeling: Give Me Some Credit](https://www.kaggle.com/code/caesarlupum/modeling-give-me-some-credit)                     | Chỉ inventory |
|     5 |  107 | [EDA — Top 100 on Leaderboard](https://www.kaggle.com/code/nicholasgah/eda-credit-scoring-top-100-on-leaderboard)        | Chỉ inventory |
|     6 |   94 | [credit-top5 solution evaluation](https://www.kaggle.com/code/bannourchaker/credit-top5-solution-evaluation-all)          | Chỉ inventory |
|     7 |   57 | [EDA, XGBoost, LightGBM &amp; SHAP](https://www.kaggle.com/code/uditnagar5/give-me-some-credit-eda-xgboost-lightgbm-shap) | Chỉ inventory |
|     8 |   53 | [Financial Distress Prediction](https://www.kaggle.com/code/prasadposture121/financial-distress-prediction)               | Chỉ inventory |
|     9 |   51 | [MLJAR AutoML](https://www.kaggle.com/code/mt77pp/mljar-automl-givemesomecredit)                                          | Chỉ inventory |
|    10 |   46 | [Starter: Give Me Some Credit](https://www.kaggle.com/code/mostig/starter-give-me-some-credit)                            | Chỉ inventory |

Vote thay đổi theo thời gian. Các số trên là **Verified đối với snapshot local**,
không phải lời khẳng định về ranking Kaggle ở mọi thời điểm.

## Kết luận ngắn

Không notebook nào trong top 3 đủ an toàn để lấy metric làm benchmark chuẩn hoặc copy nguyên pipeline.

| Notebook         | Giá trị chính                                             | Vấn đề quyết định                                                                                                      | Kết luận sử dụng                                                    |
| ---------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| #1 ScoreCard     | Minh họa đầy đủ bin → IV/WoE → LR → điểm           | Học preprocessing và target encoding trước split; tham số scale mâu thuẫn                                             | Dùng để học cấu trúc scorecard, viết lại validation             |
| #2 Starter       | Inventory rộng nhiều classifier, scaling, tuning, ensemble | Snapshot không chạy tuần tự; lọc dòng dựa vào target; đánh giá prediction trên train; metric lệch AUC           | Chỉ dùng như danh sách ý tưởng, không dùng làm implementation |
| #3 Group Project | EDA anomaly rõ, so sánh dataset/model bằng ROC-AUC        | Tiền xử lý trước CV; tuning và báo cáo trên cùng CV; không có holdout; kết luận pháp lý về`age` quá mức | Tốt nhất để học EDA, cần nested CV/holdout khi dựng lại         |

Thứ tự khuyến nghị theo giá trị cho dự án:

1. **Notebook #3** cho anomaly investigation và cách so sánh biến thể dữ liệu.
2. **Notebook #1** cho cấu trúc scorecard/WoE, nhưng phải fit mọi transformer chỉ trên train.
3. **Notebook #2** chỉ để lập checklist thuật toán; chất lượng code và provenance
   metric không đạt yêu cầu.

## Các phát hiện xuyên suốt

### 1. Vote không đồng nghĩa với độ tin cậy thực nghiệm

Top 3 đều có `execution_count = null` và không lưu output. Notebook #2 còn dừng ngay ở các cell đầu nếu chạy từ trên xuống. Vì vậy vote phản ánh mức quan tâm/hữu ích cộng đồng, không xác nhận reproducibility, validation hay chất lượng metric.

### 2. Leakage chủ yếu nằm ở preprocessing

Notebook #1 tính median, ranh giới bin, IV, WoE và chọn feature trên toàn bộ
`cs-training.csv` rồi mới split. Notebook #3 cũng tạo các dataset đã fill median,
loại/cap outlier trên toàn bộ dữ liệu trước cross-validation. Theo hướng dẫn chính
thức của scikit-learn, phải split trước và chỉ học preprocessing trên phần train;
`Pipeline` giúp giữ thống kê của fold test ra khỏi quá trình fit
([Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [Pipeline](https://scikit-learn.org/stable/modules/compose.html)).

### 3. Metric phải khớp bài toán

Competition đánh giá khả năng xếp hạng xác suất bằng ROC-AUC. Notebook #1 có tính ROC-AUC; notebook #3 dùng `scoring=['roc_auc']`; notebook #2 lại tối ưu chủ yếu weighted F1/accuracy và sau tuning còn đo prediction trên chính train. AUC phải nhận score/probability, không phải chỉ nhãn cứng ([`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)).

### 4. Model selection cần lớp đánh giá độc lập

Notebook #3 thử nhiều dataset và hàng chục cấu hình Random Forest rồi chọn AUC cao
nhất trên cùng cơ chế CV. Notebook #2 cũng grid/random search nhưng không có outer
CV hoặc holdout hợp lệ. Chọn cấu hình và báo cáo trên cùng dữ liệu tạo ước lượng
lạc quan; nested CV hoặc test set khóa riêng mới đánh giá được toàn bộ quy trình
chọn mô hình
([Nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)).

### 5. Đây không phải validation production

GiveMeSomeCredit không có cột thời gian. Không notebook nào chứng minh out-of-time stability, calibration theo kỳ, fairness, tác động kinh doanh hay khả năng áp dụng cho population mới. Kết quả Kaggle/offline chỉ là benchmark thực hành.

## Báo cáo chi tiết

- [#1 — Credit ScoreCard example](details/01-credit-scorecard-example.md)
- [#2 — Starter: credit card scoring](details/02-starter-credit-card-scoring.md)
- [#3 — Comp Stats Group Data Project](details/03-comp-stats-group-project.md)

## Nguồn và provenance

- Kaggle, [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit).
- Snapshot local:
  [`README.md`](../../../notebooks/top-voted/GiveMeSomeCredit/README.md) và
  [`manifest.json`](../../../notebooks/top-voted/GiveMeSomeCredit/manifest.json),
  truy cập 2026-07-29.
- scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html),
  truy cập 2026-07-29.
- scikit-learn, [Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
  truy cập 2026-07-29.
- scikit-learn, [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html),
  truy cập 2026-07-29.

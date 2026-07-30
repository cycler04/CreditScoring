# Kế hoạch báo cáo tổng quan bài toán và pipeline

## Mục tiêu đầu ra

Viết **một báo cáo ngắn, dễ đọc cho người mới** tại
`docs/03-problem-pipeline-notebook-overview.md`, trả lời bốn câu hỏi:

1. Credit scoring và GiveMeSomeCredit đang giải quyết bài toán gì?
2. Những điểm khó và rủi ro phương pháp quan trọng nhất là gì?
3. Pipeline local đi từ dữ liệu đến metric, scorecard và cutoff như thế nào?
4. Các notebook Kaggle được vote nhiều thường làm gì, phần nào nên học và phần nào
   không nên sao chép?

Báo cáo dự kiến khoảng **900–1.200 từ**, ưu tiên ý tưởng và luồng xử lý hơn tham số
hoặc công thức chi tiết.

## Nguồn bắt buộc

Đọc theo thứ tự sau:

1. `docs/01-kaggle-reports/competition/06-comp-give-me-some-credit.md`
   - Nguồn chính cho mục tiêu dự đoán, target, feature, class imbalance, missing và
     anomaly của GiveMeSomeCredit.
2. `docs/00-tong-quan/README.md`
   - Khung khái niệm ngắn về credit scoring, label, feature, leakage, scorecard và
     monitoring.
3. `docs/01-kaggle-reports/top-voted/overview.md`
   - Nguồn tổng hợp về top notebook: EDA, xử lý anomaly, thử nhiều model,
     WoE/scorecard, tuning, ensemble và các lỗi validation.
4. `.agents/02_architecture.md` và `src/credit_scoring/pipeline.py`
   - Nguồn sự thật cho pipeline đang được triển khai trong workspace.
5. `src/credit_scoring/data.py`
   - Nguồn sự thật cho cách pipeline local xử lý `age = 0`, mã 96/98, missing và
     extreme values.

## Nguồn chỉ mở khi cần làm rõ

- `docs/00-tong-quan/04-eda-playbook.md`: checklist EDA.
- `docs/00-tong-quan/05-modeling-playbook.md`: so sánh LR, LightGBM và WoE-LR.
- `docs/00-tong-quan/06-metrics-validation-monitoring.md`: AUC, Gini, KS và PSI.
- `docs/01-kaggle-reports/top-voted/details/01-credit-scorecard-example.md`:
  cấu trúc scorecard và lỗi leakage.
- `docs/01-kaggle-reports/top-voted/details/02-starter-credit-card-scoring.md`:
  inventory model, tuning và ensemble.
- `docs/01-kaggle-reports/top-voted/details/03-comp-stats-group-project.md`:
  EDA anomaly và lỗi model selection.

Không cần đọc lại toàn bộ notebook `.ipynb` trừ khi báo cáo tổng hợp và report chi
tiết mâu thuẫn nhau. Không dùng hai báo cáo WoE chuyên sâu làm nguồn chính vì vượt
quá độ sâu của đầu ra.

## Cấu trúc báo cáo

### 1. Problem overview

- Dự đoán xác suất một người gặp financial distress trong hai năm.
- Target là `SeriousDlqin2yrs`; dữ liệu gồm 150.000 dòng và 10 feature tín dụng.
- Đây là benchmark thực hành cho credit scoring, không phải hệ thống phê duyệt
  production.

### 2. Key points and difficulties

Chỉ giữ các ý có ảnh hưởng trực tiếp đến kết quả:

- target lệch lớp, nên accuracy không phù hợp;
- missing có thể mang tín hiệu rủi ro;
- anomaly và special code như `age = 0`, delinquency 96/98, utilization hoặc
  debt ratio cực lớn;
- leakage khi học imputation, binning, WoE hoặc feature selection trước split;
- metric phải dùng probability/ranking, chủ yếu AUC, kèm Gini và KS;
- dataset không có cột thời gian, nên split 60/20/20 hiện tại là stratified random,
  không phải out-of-time validation;
- benchmark offline không chứng minh stability, fairness, calibration hoặc hiệu
  quả production.

### 3. Overall local pipeline

Giải thích bằng một luồng duy nhất:

```text
raw CSV
  -> load và clean anomaly
  -> stratified train/valid/test 60/20/20
  -> EDA chỉ trên train
  -> raw Logistic Regression + LightGBM
  -> train-only binning, WoE/IV và WoE Logistic Regression
  -> AUC/Gini/KS
  -> scorecard 300-850, approval cutoff và PSI
  -> artifacts trong datasets/processed và outputs/
```

Chỉ giải thích vai trò của từng bước; không liệt kê hyperparameter.

### 4. What top notebooks usually do

Tổng hợp theo pattern, không tóm tắt tuần tự từng notebook:

- khám phá bad rate, missing, phân phối và anomaly;
- thử các cách impute/cap/drop và tạo nhiều phiên bản dữ liệu;
- dựng baseline rồi so sánh LR, tree model, boosting và đôi khi neural network;
- tune model, ensemble và tạo prediction;
- với scorecard: binning → WoE/IV → LR → quy đổi thành điểm và cutoff.

Kết thúc bằng nhận định: notebook tốt để học ý tưởng và checklist, nhưng vote hoặc
metric tự báo cáo không bảo đảm reproducibility; pipeline local phải giữ split và
preprocessing đúng để tránh leakage.

## Quy trình thực hiện

1. Trích mỗi nguồn thành các claim ngắn và gắn đường dẫn nguồn.
2. Đối chiếu mô tả tài liệu với code local; code là nguồn quyết định cho hành vi
   hiện tại của workspace.
3. Gom claim theo bốn phần của báo cáo, loại nội dung lặp và chi tiết WoE không cần
   thiết.
4. Viết bản đầu trong giới hạn 900–1.200 từ.
5. Soát mọi claim định lượng, thuật ngữ split và ranh giới giữa benchmark với
   production.
6. Kiểm tra link nội bộ và chạy `uv run python .agents/scripts/01_validate_workspace.py --full`.

## Tiêu chí hoàn thành

- Chỉ có một báo cáo chính, đọc được độc lập.
- Người mới hiểu được bài toán và luồng pipeline mà không cần đọc code.
- Phần notebook mô tả pattern chung và cảnh báo phương pháp, không biến thành danh
  sách review dài.
- Không gọi random split là out-of-time; không suy diễn hiệu quả production.
- Mọi claim quan trọng truy vết được về report, code hoặc artifact trong workspace.

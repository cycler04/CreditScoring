# Pre-Sprint 0 — Credit Scoring | Tổng quan

Mục tiêu tuần (từ [notes/task.txt](../../notes/task.txt)):

1. Có knowledge base cơ bản về cho vay và credit scoring.
2. Biết label và feature thường đến từ đâu.
3. Tìm hiểu trước phương pháp EDA và xây mô hình ML khi có dữ liệu.

## Đọc theo thứ tự

| # | File | Trả lời mục tiêu |
|---|------|------------------|
| 1 | [01-co-ban-cho-vay-va-credit-scoring.md](01-co-ban-cho-vay-va-credit-scoring.md) | MT1 — nghiệp vụ cho vay, PD/LGD/EAD, scorecard, Basel |
| 2 | [02-label-target-den-tu-dau.md](02-label-target-den-tu-dau.md) | MT2 — định nghĩa default, observation/performance window, vintage |
| 3 | [03-feature-den-tu-dau.md](03-feature-den-tu-dau.md) | MT2 — nguồn feature: application, bureau, behavioral, alternative |
| 4 | [04-eda-playbook.md](04-eda-playbook.md) | MT3 — quy trình EDA cho dữ liệu tín dụng |
| 5 | [05-modeling-playbook.md](05-modeling-playbook.md) | MT3 — WoE+LR scorecard vs GBDT, pipeline chuẩn |
| 6 | [06-metrics-validation-monitoring.md](06-metrics-validation-monitoring.md) | MT3 — AUC/Gini/KS/IV/PSI, cutoff, giám sát drift |
| 7 | [07-ke-hoach-tuan-checklist.md](07-ke-hoach-tuan-checklist.md) | Kế hoạch 5 ngày + tiêu chí "xong" |

Report chi tiết từng link Kaggle: [../10-kaggle-reports/](../10-kaggle-reports/README.md)

## Tóm tắt 10 dòng (nếu chỉ đọc 1 thứ)

- Credit scoring = ước lượng **PD** (probability of default) của 1 khách tại thời điểm quyết định cho vay.
- Label **không có sẵn**, phải tự định nghĩa: "default" = 90+ ngày quá hạn (chuẩn Basel), quan sát trong performance window 12–24 tháng sau giải ngân.
- Feature đến từ 4 lớp: hồ sơ vay (application), lịch sử tín dụng ngoài (credit bureau), hành vi nội bộ (previous loans, POS, credit card, installments), và alternative data.
- Dữ liệu luôn **imbalanced** (bad rate thường 2–10%) — dùng AUC/Gini, không dùng accuracy.
- Bẫy lớn nhất: **leakage** — feature chỉ tồn tại sau khi khoản vay đã chạy (recoveries, last_payment_amount, số ngày quá hạn hiện tại) sẽ cho AUC 0.95+ giả.
- Hai trường phái mô hình: **WoE + Logistic Regression** (scorecard, giải thích được, chịu được audit) và **GBDT/LightGBM** (AUC cao hơn, khó giải thích).
- Sản phẩm cuối không phải xác suất mà là **điểm số** (ví dụ 300–900) + **cutoff** để duyệt/từ chối.
- Mô hình phải **ổn định theo thời gian**: đo bằng PSI (drift) và gini theo tuần/tháng — đây chính là chủ đề cuộc thi Home Credit Model Stability.
- Split theo **thời gian** (out-of-time), không random, nếu không sẽ đánh giá lạc quan.
- Tuần này chưa có dữ liệu thật → mục tiêu là dựng sẵn checklist EDA + template pipeline, chạy thử trên GiveMeSomeCredit (nhẹ, 150k dòng, 10 feature).

## Nguồn

Toàn bộ nội dung trong folder này tổng hợp từ 7 link trong task.txt (xem report chi tiết) + kiến thức chuẩn ngành credit risk. Chỗ nào là con số lấy trực tiếp từ trang Kaggle đều được ghi rõ trong report chi tiết.

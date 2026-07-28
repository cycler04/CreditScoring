# 7. Kế hoạch tuần Pre-Sprint 0 + checklist

Chưa có dữ liệu thật → tuần này sản phẩm là **kiến thức + template chạy được**, không phải mô hình.

## Kế hoạch 5 ngày

| Ngày | Việc | Verify (tiêu chí xong) |
|---|---|---|
| 1 | Đọc [01](01-co-ban-cho-vay-va-credit-scoring.md), [02](02-label-target-den-tu-dau.md). Đọc report cuộc thi [GiveMeSomeCredit](../10-kaggle-reports/06-comp-give-me-some-credit.md) và [Home Credit Default Risk](../10-kaggle-reports/07-comp-home-credit-default-risk.md) | Viết được định nghĩa default + performance window cho sản phẩm của mình, bằng lời, không nhìn tài liệu |
| 2 | Đọc [03](03-feature-den-tu-dau.md) + report [notebook Gentle Introduction](../10-kaggle-reports/01-nb-start-here-gentle-introduction.md). Tải GiveMeSomeCredit (14 MB), chạy EDA theo [04](04-eda-playbook.md) | Có notebook EDA riêng: phân phối target, missing, anomaly, bad rate theo decile của 10 biến |
| 3 | Baseline model trên GiveMeSomeCredit: LogisticRegression + LightGBM | Có AUC của cả hai. LightGBM nên ~0.86 trên bộ này |
| 4 | Đọc [05](05-modeling-playbook.md) + report [WoE & Scorecard](../10-kaggle-reports/04-nb-credit-risk-eda-woe-scorecard-2.md). Tự viết hàm `woe_iv(df, col, target)` và `bin_by_tree()` | Bảng IV cho 10 biến GiveMeSomeCredit; WoE đơn điệu sau khi gộp bin |
| 5 | Dựng scorecard 300–850 từ LR trên WoE. Đọc [06](06-metrics-validation-monitoring.md) + report [Model Stability](../10-kaggle-reports/05-comp-home-credit-model-stability.md). Viết hàm `psi()` | Có bảng scorecard (feature, bin, điểm). PSI tính được giữa 2 tập bất kỳ |

Vì sao chọn GiveMeSomeCredit để tập tay: 150,000 dòng, 10 feature, một file 7 MB, chạy trên laptop trong vài giây. Home Credit Default Risk 2.68 GB / 10 bảng / 346 cột — để dành khi đã vững quy trình.

## Sản phẩm bàn giao cuối tuần

1. `docs/` này (xong).
2. Notebook EDA trên GiveMeSomeCredit.
3. Module tiện ích tái dùng được: `woe_iv()`, `bin_by_tree()`, `psi()`, `gini_by_period()`, `scorecard_from_lr()`.
4. Danh sách câu hỏi cần hỏi bộ phận nghiệp vụ (mục dưới).

## Trạng thái phần đã tự động hóa

Pipeline chạy lại bằng `./scripts/run_all.sh`; cấu trúc và cách chạy xem tại
[README dự án](../../README.md).

- [x] Tải và kiểm tra GiveMeSomeCredit: 150.000 dòng, 10 feature.
- [x] EDA trên training split: target, missing, anomaly và bad rate decile của
  10 biến; có [notebook](../../outputs/eda/give_me_some_credit_eda.ipynb).
- [x] Baseline Logistic Regression và LightGBM; kết quả lần chạy hiện tại ở
  [metrics.csv](../../outputs/models/metrics.csv).
- [x] WoE/IV cho 10 biến; tối đa 6 bin số + 1 bin missing và WoE đơn điệu.
- [x] Scorecard integer 300–850, cutoff approval 60%/70%/80% và score PSI.
- [x] Các hàm tái sử dụng nằm trong `src/credit_scoring/`; kiểm thử bằng
  `./scripts/check.sh`.

Giới hạn: GiveMeSomeCredit không có cột thời gian nên lần chạy này dùng
stratified random split 60/20/20, không phải out-of-time split.
`gini_by_period()` đã có và được unit test nhưng cần dữ liệu thật có cột thời
gian để sinh báo cáo thực tế.

## Câu hỏi phải hỏi nghiệp vụ / data owner trước Sprint 1

**Về label**
- [ ] Sản phẩm cho vay là gì? Kỳ hạn bao lâu? (quyết định performance window)
- [ ] Định nghĩa default hiện hành của tổ chức? Có văn bản không?
- [ ] Bảng nào chứa DPD / lịch trả nợ? DPD tính sẵn hay tự tính?
- [ ] Nợ tái cơ cấu, bán nợ, xóa nợ được đánh dấu ở đâu?
- [ ] Fraud tách riêng được không?

**Về feature**
- [ ] Có nối được CIC / bureau không? Độ trễ dữ liệu bao nhiêu?
- [ ] Dữ liệu nào **có tại thời điểm duyệt** vs chỉ có sau đó? Cần bảng đối chiếu.
- [ ] Có snapshot lịch sử không, hay bảng bị ghi đè (SCD type 1)? Nếu bị ghi đè thì không tái tạo được trạng thái quá khứ → leakage không tránh khỏi.
- [ ] Biến nào bị cấm dùng vì lý do pháp lý?

**Về vận hành**
- [ ] Mô hình chấm điểm realtime hay batch?
- [ ] Approval rate hiện tại? Bad rate hiện tại?
- [ ] Đang dùng rule engine hay scorecard cũ? Baseline để so là gì?
- [ ] Chi phí một khoản bad ≈ bao nhiêu? Lợi nhuận một khoản good? (để tính cutoff)

## Checklist kiến thức — tự kiểm tra

Đánh dấu khi giải thích được cho người khác trong 2 phút, không nhìn tài liệu:

- [ ] EL = PD × LGD × EAD, mỗi thành phần nghĩa gì
- [ ] Application vs behavioral scorecard khác gì
- [ ] Observation window vs performance window
- [ ] Vì sao default = 90+ DPD
- [ ] Vì sao dùng AUC chứ không dùng accuracy
- [ ] Gini = 2·AUC − 1
- [ ] WoE tính thế nào, IV tính thế nào, ngưỡng IV
- [ ] Vì sao WoE cần đơn điệu
- [ ] Score = Base + Factor·ln(odds), PDO là gì
- [ ] PSI tính thế nào, ngưỡng 0.1 / 0.25
- [ ] Vì sao phải split out-of-time
- [ ] Kể được 5 feature bị leak điển hình trong dữ liệu cho vay
- [ ] Công thức stability metric của Home Credit và vì sao hệ số 88
- [ ] Depth 0 / 1 / 2 nghĩa là gì và aggregate thế nào
- [ ] Reject inference là vấn đề gì

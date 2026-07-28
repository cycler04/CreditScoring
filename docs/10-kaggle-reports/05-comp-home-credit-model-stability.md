# Report #5 — Home Credit: Credit Risk Model Stability

**Link:** https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability
**Host:** Home Credit Group · Featured Code Competition
**Thời gian:** 5/2/2024 – 27/5/2024
**Giải thưởng:** $105,000
**Quy mô:** 27,479 entrants · 5,315 participants · 3,856 teams · 91,975 submissions
**Tag:** Tabular, Banking, Custom Metric

> Ghi chú nguồn: fetch được nguyên văn trang Overview / Evaluation / Data. Mọi trích dẫn và công thức dưới đây là nguyên văn.

## Vì sao đây là link quan trọng nhất trong task.txt

Sáu link kia dạy cách **xây** mô hình. Link này dạy cách mô hình **hỏng** — và hỏng như thế nào theo thời gian. Đó là vấn đề số một của credit scoring trong sản xuất, và là thứ Kaggle thường bỏ qua.

## Bối cảnh (nguyên văn, đáng đọc kỹ)

> "In the real world, clients' behaviors change constantly, so every scorecard must be updated regularly, which takes time. The scorecard's stability in the future is critical, as a sudden drop in performance means that loans will be issued to worse clients on average. The core of the issue is that loan providers aren't able to spot potential problems any sooner than the first due dates of those loans are observable. Given the time it takes to redevelop, validate, and implement the scorecard, stability is highly desirable. There is a trade-off between the stability of the model and its performance, and a balance must be reached before deployment."

Bốn ý phải nhớ:
1. Hành vi khách hàng thay đổi liên tục → scorecard phải cập nhật định kỳ.
2. Suy giảm hiệu năng = đang cho vay khách xấu hơn, và **chỉ phát hiện được sau khi kỳ trả nợ đầu tiên đến hạn**.
3. Từ lúc phát hiện tới lúc thay mô hình mới còn cả quá trình redevelop + validate + implement.
4. Có **trade-off giữa stability và performance** — phải cân bằng trước khi deploy.

Bối cảnh kinh doanh: Home Credit thành lập 1997, cho vay tiêu dùng, tập trung vào người **ít hoặc không có lịch sử tín dụng**. Đây là lý do tồn tại của toàn bộ bài toán: không có bureau data thì phải tìm signal ở nơi khác.

## Metric — gini stability (nguyên văn)

```
gini = 2 × AUC − 1
```

Quy trình tính:
1. Tính gini cho các dự đoán ứng với **từng `WEEK_NUM`**.
2. Fit hồi quy tuyến tính `a·x + b` qua chuỗi gini theo tuần.
3. `falling_rate = min(0, a)` — dùng để phạt mô hình suy giảm khả năng dự báo.
4. Tính std của **phần dư** quanh đường hồi quy — phạt mô hình dao động.

```
stability metric = mean(gini) + 88.0 × min(0, a) − 0.5 × std(residuals)
```

### Đọc công thức

| Thành phần | Vai trò | Hệ số |
|---|---|---|
| `mean(gini)` | Hiệu năng trung bình | +1 |
| `min(0, a)` | Độ dốc suy giảm (chỉ phạt khi âm) | **×88** |
| `std(residuals)` | Dao động quanh xu hướng | ×(−0.5) |

Hệ số 88 là tuyên bố rất mạnh. Nếu gini giảm 0.002 mỗi tuần, phạt = 88 × 0.002 = **0.176** — lớn hơn phần lớn khoảng cách hiệu năng giữa các mô hình. Nói cách khác: cuộc thi này thà lấy mô hình gini 0.55 phẳng còn hơn mô hình gini 0.60 tụt dần.

Chi tiết tinh tế: `min(0, a)` nghĩa là mô hình **tốt lên** theo thời gian không được thưởng, chỉ không bị phạt. Bất đối xứng có chủ ý.

Std phần dư chỉ nhân 0.5 → nhiễu tuần-qua-tuần được tha thứ nhiều hơn suy giảm có hệ thống. Hợp lý: nhiễu do cỡ mẫu, suy giảm do mô hình lỗi thời.

**Áp dụng lại được cho dự án nội bộ** mà không cần dữ liệu Home Credit — chỉ cần một cột thời gian và đủ mẫu mỗi kỳ. Nên đưa vào bộ công cụ monitoring.

## Stability Prize — hạng mục riêng

Có giải riêng $10,000 cho stability, yêu cầu qualification (nguyên văn):

> - Your solution is among top 20% on the private leaderboard
> - Your approach to stability should be: 1) general, not dependent explicitly on used dataset, and 2) **incorporate stability directly into your training model / loss function** (not on the level of feature preparation or feature selection)
> - Your notebook is public

Chấm theo: Innovation 40%, Quality 20%, Clarity 20%, Generalization 20%.

Điểm đáng chú ý: yêu cầu đưa stability vào **loss function**, không chỉ vào bước chọn feature. Đây là hướng nghiên cứu thật sự — phần lớn team chỉ xử lý stability bằng cách loại feature drift, còn đưa nó vào mục tiêu tối ưu thì khó hơn nhiều.

## Cấu trúc dữ liệu

### Khái niệm depth (nguyên văn)

> - **depth=0** — static features directly tied to a specific `case_id`
> - **depth=1** — each `case_id` has an associated historical record, indexed by `num_group1`
> - **depth=2** — each `case_id` has an associated historical record, indexed by both `num_group1` and `num_group2`

Với depth > 0 phải aggregate lịch sử về một feature duy nhất cho mỗi `case_id`. Khi `num_groupN` đại diện chỉ số người thì **index 0 là chính người nộp đơn**.

### Các nhóm bảng

| Nhóm bảng | Depth | Nguồn |
|---|---|---|
| `static_cb_0` | 0 | external |
| `applprev_1`, `applprev_2` | 1, 2 | internal — hồ sơ vay trước |
| `person_1`, `person_2` | 1, 2 | internal |
| `other_1` | 1 | internal |
| `deposit_1` | 1 | internal — tiền gửi |
| `debitcard_1` | 1 | internal — thẻ ghi nợ |
| `tax_registry_a_1`, `_b_1`, `_c_1` | 1 | external — 3 nhà cung cấp đăng ký thuế |
| `credit_bureau_a_1`, `_a_2` | 1, 2 | external — bureau A |
| `credit_bureau_b_1`, `_b_2` | 1, 2 | external — bureau B |

Bảng lớn được chia nhỏ theo `WEEK_NUM` để giới hạn kích thước file (`credit_bureau_a_2` có 11 file train, 12 file test).

Ba nhóm feature đáng chú ý so với Home Credit Default Risk 2018: **tax registry** (dữ liệu thuế → xác minh thu nhập), **deposit**, **debitcard**. Đây là "alternative data" cho nhóm khách không có lịch sử tín dụng.

### Cột đặc biệt (nguyên văn)

| Cột | Ý nghĩa |
|---|---|
| `case_id` | Khóa duy nhất mỗi hồ sơ tín dụng, dùng để join |
| `date_decision` | Ngày ra quyết định duyệt khoản vay — **chính là T0** |
| `WEEK_NUM` | Số tuần dùng để tổng hợp; trong tập test tiếp tục nối tiếp từ giá trị cuối của train |
| `MONTH` | Tháng, dùng để tổng hợp |
| `target` | Xác định sau một khoảng thời gian, dựa trên việc khách có default hay không |
| `num_group1`, `num_group2` | Chỉ số bản ghi lịch sử cho depth 1 và 2 |

`WEEK_NUM` trong test **nối tiếp** train — nghĩa là test hoàn toàn là tương lai. Đây là out-of-time split ở mức thiết kế cuộc thi.

### Quy ước đặt tên predictor (nguyên văn)

Hậu tố chữ hoa cho biết loại biến đổi đã áp dụng:

| Ký hiệu | Nghĩa |
|---|---|
| **P** | Transform DPD (days past due) |
| **M** | Masking categories |
| **A** | Transform amount |
| **D** | Transform date |
| **T** | Unspecified transform |
| **L** | Unspecified transform |

Ví dụ: `maxdbddpdtollast6m_4187119P` — biến DPD, max, cửa sổ 6 tháng gần nhất.

Định nghĩa đầy đủ nằm trong `feature_definitions.csv`.

**Đáng copy:** quy ước đặt tên có hậu tố loại biến giúp xử lý hàng loạt (`df.filter(regex='P$')` để lấy mọi biến DPD). Với ~470 predictor thì đây không phải trang trí mà là điều kiện sống còn.

## Cảnh báo trong mô tả dữ liệu (nguyên văn)

> "some external data providers might not be available for future (test) evaluations, which is anticipated"

Nhà cung cấp dữ liệu ngoài **có thể biến mất** ở kỳ đánh giá tương lai. Đây là nguyên nhân thực tế gây mất stability, và là lý do phải có phương án fallback cho mọi nguồn external trong thiết kế mô hình sản xuất.

## Điều kiện Code Competition

- CPU notebook ≤ 12 giờ chạy
- GPU notebook ≤ 12 giờ chạy
- **Tắt internet**
- Cho phép dữ liệu ngoài và pre-trained model công khai
- File nộp phải tên `submission.csv`, format `case_id,score`

Ràng buộc 12 giờ + tắt internet trên bộ dữ liệu hàng chục GB là bài toán kỹ thuật riêng: phần lớn team dùng **Polars** thay pandas và parquet thay csv để nạp/aggregate kịp.

## Rút ra cho dự án

**Copy được ngay:**
1. **Công thức stability metric** — đưa vào bộ monitoring: tính gini theo tuần/tháng, fit đường xu hướng, phạt độ dốc âm và độ dao động.
2. **Khung depth 0/1/2** — cách tư duy chuẩn cho dữ liệu tín dụng nhiều bảng.
3. **Quy ước đặt tên predictor có hậu tố loại** (P/M/A/D/T/L) hoặc tương đương.
4. `date_decision` làm mốc cắt cho mọi feature.
5. Test là tương lai của train — thiết kế validation phải giống production.
6. Danh mục nguồn alternative data cần khảo sát: đăng ký thuế, tiền gửi, thẻ ghi nợ.

**Nguyên tắc rút ra:**
- Mô hình ổn định > mô hình mạnh, khi chênh lệch hiệu năng nhỏ.
- Mọi nguồn external phải có phương án khi nó chết.
- Đo hiệu năng theo **từng kỳ thời gian**, không chỉ một con số tổng.
- Cân nhắc đưa stability vào chính hàm mục tiêu, không chỉ vào bước chọn feature.

## Trích dẫn

Daniel Herman, Tomas Jelinek, Walter Reade, Maggie Demkin, and Addison Howard. *Home Credit - Credit Risk Model Stability.* https://kaggle.com/competitions/home-credit-credit-risk-model-stability, 2024. Kaggle.

## Liên quan
- Cuộc thi tiền nhiệm: [Report #7 — Home Credit Default Risk](07-comp-home-credit-default-risk.md)
- Monitoring và PSI: [06-metrics-validation-monitoring.md](../00-tong-quan/06-metrics-validation-monitoring.md)

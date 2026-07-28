# 3. Feature đến từ đâu

## 3.1 Bốn lớp nguồn dữ liệu

| Lớp | Nguồn | Ví dụ | Có ở link nào |
|---|---|---|---|
| **A. Application** | Form khách khai + hệ thống duyệt | tuổi, thu nhập, nghề, số tiền vay, kỳ hạn, mục đích | tất cả |
| **B. Credit bureau** | CIC / bureau bên ngoài | số khoản vay đang có, dư nợ, DPD lịch sử, số lần truy vấn | HCDR `bureau.csv`, Model Stability `credit_bureau_a/b` |
| **C. Behavioral (nội bộ)** | Lịch sử giao dịch với chính tổ chức | khoản vay trước, lịch sử trả góp, sao kê thẻ | HCDR `previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance` |
| **D. Alternative** | Nguồn khác | thuế, telco, tiền gửi, thẻ ghi nợ, device/geo | Model Stability `tax_registry_a/b/c`, `deposit_1`, `debitcard_1` |

Home Credit - Credit Risk Model Stability đánh dấu rõ từng bảng là `internal data source` hay `external data source`, và cảnh báo: *"some external data providers might not be available for future (test) evaluations"* — tức nguồn ngoài có thể **biến mất** trong production. Đây là một nguyên nhân trực tiếp gây mất stability.

## 3.2 Cấu trúc quan hệ: depth 0 / 1 / 2

Model Stability định nghĩa gọn nhất, dùng làm khung tư duy chung:

- **depth = 0** — feature tĩnh, gắn trực tiếp với `case_id`. Dùng thẳng làm feature.
- **depth = 1** — mỗi `case_id` có nhiều bản ghi lịch sử, đánh chỉ số bằng `num_group1`.
- **depth = 2** — lịch sử của lịch sử, đánh chỉ số bằng `num_group1` + `num_group2`.

Với depth > 0 phải **aggregate** về một dòng / một `case_id`. Quy ước hữu ích: `num_groupN = 0` là chính người nộp đơn (applicant), các index khác là người liên quan.

Home Credit Default Risk là cùng ý tưởng nhưng đặt tên theo nghiệp vụ:

```
application_train.csv  (1 dòng = 1 hồ sơ, có TARGET)   ← depth 0
├── bureau.csv                    (n khoản vay ngoài)   ← depth 1
│   └── bureau_balance.csv        (n tháng mỗi khoản)   ← depth 2
├── previous_application.csv      (n hồ sơ cũ)          ← depth 1
│   ├── POS_CASH_balance.csv      (n tháng)             ← depth 2
│   ├── credit_card_balance.csv   (n tháng)             ← depth 2
│   └── installments_payments.csv (n lần trả)           ← depth 2
```
Tổng 10 file, 346 cột, 2.68 GB (số liệu từ trang Data của cuộc thi).

## 3.3 Aggregation — nơi phần lớn signal nằm

Với bảng depth ≥ 1, mẫu chuẩn:

```python
agg = (child.groupby('case_id')['amount']
             .agg(['min','max','mean','sum','std','count'])
             .add_prefix('bureau_amount_'))
base = base.merge(agg, on='case_id', how='left')
```

Các trục aggregate nên nghĩ tới:
- **Thống kê**: min / max / mean / sum / std / count / nunique.
- **Cửa sổ thời gian**: 3m / 6m / 12m / all-time (`maxdpdlast6m` kiểu Home Credit).
- **Recency**: giá trị của bản ghi gần nhất, số ngày kể từ bản ghi gần nhất.
- **Trend**: tỷ số cửa sổ ngắn / cửa sổ dài (ví dụ `dpd_6m / dpd_12m` > 1 = đang xấu đi).
- **Tỷ lệ**: số kỳ trả trễ / tổng số kỳ.

Đặt tên cột theo quy ước từ đầu (`{bảng}_{cột}_{hàm}_{cửa sổ}`), nếu không sau vài trăm feature sẽ không truy được nguồn gốc.

## 3.4 Ratio features — rẻ và mạnh

Feature thô ít nói lên gánh nặng nợ; **tỷ lệ** mới nói. Đây là nhóm feature domain-knowledge kinh điển của Home Credit Default Risk:

```python
CREDIT_INCOME_PERCENT  = AMT_CREDIT   / AMT_INCOME_TOTAL   # vay gấp mấy lần thu nhập
ANNUITY_INCOME_PERCENT = AMT_ANNUITY  / AMT_INCOME_TOTAL   # DTI — gánh nặng trả nợ
CREDIT_TERM            = AMT_ANNUITY  / AMT_CREDIT         # kỳ hạn ngầm định
DAYS_EMPLOYED_PERCENT  = DAYS_EMPLOYED / DAYS_BIRTH        # % đời đi làm
```

Trong Give Me Some Credit, hai biến mạnh nhất cũng chính là tỷ lệ: `RevolvingUtilizationOfUnsecuredLines` (dư nợ / hạn mức) và `DebtRatio` (nghĩa vụ trả nợ hàng tháng / thu nhập).

Notebook LendingClub cũng tạo ratio từ cặp biến tương quan cao thay vì bỏ một biến: `loan_amnt_div_instlmnt = loan_amnt / installment`.

## 3.5 Feature từ ngày tháng

Ngày tháng thô vô dụng, hiệu số mới có nghĩa. Mẫu từ notebook LendingClub:

```python
loan_age              = today - issue_d
credit_history_length = issue_d - earliest_cr_line     # độ dài lịch sử tín dụng
time_since_last_payment     = today - last_pymnt_d
time_since_last_credit_pull = today - last_credit_pull_d
recent_payment = (time_since_last_payment <= 30).astype(int)
```

Home Credit Default Risk mã hóa sẵn kiểu này: `DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION` là số âm tính ngược từ ngày nộp hồ sơ. `DAYS_BIRTH / -365` = tuổi.

**Cảnh báo:** dùng `today` (ngày chạy code) như notebook LendingClub là sai về mặt sản xuất — mốc phải là `date_decision` / T0 của từng hồ sơ, nếu không thì feature phụ thuộc vào lúc bạn chạy script và rò rỉ tương lai.

## 3.6 Mã hóa biến phân loại

| Cách | Khi nào | Ghi chú |
|---|---|---|
| Label encoding | Biến nhị phân (≤ 2 giá trị) | Không tạo thứ tự giả |
| One-hot | Cardinality thấp (< ~15) | Nhớ `align()` train/test để khớp cột |
| **WoE encoding** | Scorecard | Chuẩn ngành, xem [05-modeling-playbook.md](05-modeling-playbook.md) |
| Gộp nhóm hiếm | Category tần suất thấp | Notebook LendingClub gộp `ANY`/`MORTGAGE`/`NONE` → `other` dựa trên WoE gần nhau |
| Bỏ | Cardinality quá cao, IV thấp | LendingClub bỏ `addr_state` (51 giá trị), `purpose`, `emp_title`, `zip_code` |

LightGBM/CatBoost xử lý categorical trực tiếp — không cần one-hot khi dùng GBDT.

## 3.7 Feature CẤM dùng (leakage)

Đây là mục quan trọng nhất của file này.

Một feature bị leak nếu nó **chỉ tồn tại/thay đổi sau thời điểm quyết định T0**, hoặc nó là hệ quả của chính label.

Ví dụ leak rõ ràng trong tập LendingClub (notebook #1 của Beata Faron giữ lại và chúng lọt vào top feature importance):

| Cột | Vì sao leak |
|---|---|
| `recoveries`, `collection_recovery_fee` | Chỉ khác 0 khi khoản vay **đã** charged-off |
| `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` | Tổng đã thu — tích lũy suốt đời khoản vay |
| `last_pymnt_amnt`, `time_since_last_payment` | Hành vi trả nợ sau giải ngân |
| `out_prncp` | Dư nợ còn lại hiện tại |
| `debt_settlement_flag` | Cờ đã thỏa thuận xử lý nợ |
| `loan_age` | Khoản vay càng già càng có cơ hội xấu |

Đây là lý do notebook đó báo AUC rất cao (mục ROC-AUC ghi *"0.9 - 1.0 → Excellent"*). Với application scorecard, AUC thực tế của mô hình tốt nằm khoảng **0.70–0.80** (Home Credit Default Risk: notebook baseline 0.75, top LB ~0.80). Thấy AUC > 0.9 trên bài credit scoring → nghi leakage trước, mừng sau.

Nhóm feature khác cần cân nhắc: **điểm số từ mô hình khác**. Notebook LendingClub cố tình bỏ `fico_range_low/high`, `last_fico_range_low/high` vì đó là điểm tín dụng ngoài, dùng thì mô hình chỉ học lại FICO. Ngược lại, Home Credit Default Risk giữ `EXT_SOURCE_1/2/3` (điểm từ nguồn ngoài, đã chuẩn hóa) và đó là 3 biến mạnh nhất bộ dữ liệu. Quyết định giữ hay bỏ là quyết định nghiệp vụ: nguồn đó có sẵn tại thời điểm chấm điểm trong production không?

## 3.8 Checklist feature khi có dữ liệu thật

- [ ] Liệt kê mọi bảng có thể join, kèm khóa và độ trễ cập nhật (data latency).
- [ ] Với mỗi cột: giá trị này được ghi vào hệ thống **lúc nào**? Trước hay sau T0?
- [ ] Nguồn ngoài nào có thể mất trong tương lai? Có phương án fallback?
- [ ] Xác định quy ước đặt tên feature trước khi sinh feature.
- [ ] Đặt trần số feature cho vòng đầu (30–80 là đủ cho scorecard), tránh sinh 3,000 feature rồi ngập.

# Report #7 — Home Credit Default Risk

**Link:** https://www.kaggle.com/competitions/home-credit-default-risk
**Host:** Home Credit Group · Featured Prediction Competition · 2018
**Giải thưởng:** $70,000
**Dữ liệu:** 2.68 GB · 10 file · **346 cột**
**Metric:** AUC
**Câu hỏi đề bài (nguyên văn):** *"Can you predict how capable each applicant is of repaying a loan?"*

> Ghi chú nguồn: phần Dataset Description dưới đây lấy nguyên văn từ trang Data của cuộc thi.

## Vì sao đây là bộ dữ liệu tham chiếu quan trọng nhất

Đây là bộ dữ liệu credit scoring công khai **giống production nhất**: nhiều bảng, quan hệ một-nhiều nhiều tầng, dữ liệu nội bộ trộn dữ liệu bureau, missing nhiều, có anomaly mã hóa. Ai làm chủ được bộ này là làm chủ được phần kỹ thuật dữ liệu của một dự án credit scoring thật.

Hai trong bốn notebook của task.txt ([#1](01-nb-start-here-gentle-introduction.md), [#2](02-nb-complete-eda-feature-importance.md)) dùng bộ này.

## Cấu trúc 7 bảng (nguyên văn từ trang Data)

**application_{train|test}.csv**
> Bảng chính, tách làm hai file Train (có TARGET) và Test (không có TARGET). Dữ liệu tĩnh cho mọi hồ sơ. Một dòng = một khoản vay.

**bureau.csv**
> Toàn bộ khoản vay trước đây của khách tại các tổ chức tài chính khác, đã được báo lên Credit Bureau. Với mỗi khoản vay trong mẫu, có bấy nhiêu dòng bằng số khoản vay khách từng có ở Credit Bureau trước ngày nộp hồ sơ.

**bureau_balance.csv**
> Số dư hàng tháng của các khoản vay trước tại Credit Bureau. Một dòng cho mỗi tháng lịch sử của mỗi khoản vay.

**POS_CASH_balance.csv**
> Ảnh chụp số dư hàng tháng của các khoản vay POS (point of sales) và vay tiền mặt trước đây mà khách có tại Home Credit.

**credit_card_balance.csv**
> Ảnh chụp số dư hàng tháng của các thẻ tín dụng trước đây khách có tại Home Credit.

**previous_application.csv**
> Toàn bộ hồ sơ vay Home Credit trước đây của khách. Một dòng cho mỗi hồ sơ cũ.

**installments_payments.csv**
> Lịch sử trả nợ cho các khoản đã giải ngân tại Home Credit. Gồm (a) một dòng cho mỗi lần thanh toán, cộng (b) một dòng cho mỗi lần bỏ lỡ thanh toán.

**HomeCredit_columns_description.csv** — mô tả các cột.

### Sơ đồ quan hệ

```
application_train.csv    (SK_ID_CURR, 307,511 dòng, 122 cột, có TARGET)
│
├── bureau.csv                      SK_ID_CURR → SK_ID_BUREAU   (depth 1)
│   └── bureau_balance.csv          SK_ID_BUREAU × tháng        (depth 2)
│
└── previous_application.csv        SK_ID_CURR → SK_ID_PREV     (depth 1)
    ├── POS_CASH_balance.csv        SK_ID_PREV × tháng          (depth 2)
    ├── credit_card_balance.csv     SK_ID_PREV × tháng          (depth 2)
    └── installments_payments.csv   SK_ID_PREV × lần trả        (depth 2)
```

Đây chính là mô hình depth 0/1/2 mà cuộc thi 2024 định nghĩa thành thuật ngữ chính thức — xem [Report #5](05-comp-home-credit-model-stability.md).

Chú ý phân biệt hai nhánh:
- Nhánh **bureau** = lịch sử tín dụng ở **tổ chức khác**.
- Nhánh **previous_application** = lịch sử với **chính Home Credit** (behavioral nội bộ).

Hai nguồn này khác nhau về độ tin cậy, độ trễ, và khả năng có được trong production. Trong dự án thật phải hỏi rõ mình có nhánh nào.

## Target

`TARGET` ∈ {0, 1}, 1 = khách có **payment difficulties** (trễ hơn X ngày ở một trong Y kỳ trả đầu tiên). Bad rate ~**8.07%** (24,825 / 307,511).

## Đặc điểm dữ liệu cần biết

### Nhóm feature trong application_train (122 cột)

| Nhóm | Ví dụ cột |
|---|---|
| Hợp đồng | `NAME_CONTRACT_TYPE`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` |
| Nhân khẩu | `CODE_GENDER`, `DAYS_BIRTH`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS` |
| Việc làm / thu nhập | `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `OCCUPATION_TYPE`, `DAYS_EMPLOYED`, `ORGANIZATION_TYPE` |
| Nhà ở | `NAME_HOUSING_TYPE`, 47 cột `*_AVG` / `*_MODE` / `*_MEDI` |
| **Điểm nguồn ngoài** | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| Giấy tờ | `FLAG_DOCUMENT_2` … `FLAG_DOCUMENT_21` |
| Liên hệ | `FLAG_MOBIL`, `FLAG_EMAIL`, `DAYS_LAST_PHONE_CHANGE` |
| Vòng xã hội | `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE` |
| Truy vấn bureau | `AMT_REQ_CREDIT_BUREAU_HOUR` … `_YEAR` |

`EXT_SOURCE_1/2/3` là ba biến mạnh nhất — điểm chuẩn hóa từ nguồn dữ liệu ngoài. Bài học nghiệp vụ: **điểm từ bên thứ ba thường mạnh hơn mọi feature tự xây**, nhưng phụ thuộc nhà cung cấp (xem cảnh báo ở [Report #5](05-comp-home-credit-model-stability.md) về nguồn external có thể biến mất).

Nhóm 47 cột nhà ở missing ~50–70% và chia làm 3 biến thể `_AVG` / `_MODE` / `_MEDI` tương quan gần như hoàn toàn với nhau — chỉ nên giữ một biến thể.

### Anomaly nổi tiếng

- `DAYS_EMPLOYED = 365243` ở hơn 55,000 dòng — mã sentinel cho nhóm không đi làm. Xử lý: tạo cờ rồi thay NaN. Chi tiết ở [Report #1](01-nb-start-here-gentle-introduction.md).
- `DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH` đều **âm**, tính ngược từ ngày nộp hồ sơ.
- `CODE_GENDER` có giá trị `XNA` (4 dòng).
- `AMT_INCOME_TOTAL` max = 117,000,000 — outlier một dòng.
- `NAME_FAMILY_STATUS` có `Unknown`.

## Mức hiệu năng tham chiếu

| Cách làm | AUC public LB |
|---|---|
| Logistic Regression, chỉ `application_train` | ~0.67 |
| Random Forest, chỉ `application_train` | ~0.68 |
| LightGBM, chỉ `application_train` | ~0.75 |
| LightGBM + aggregate đủ 6 bảng phụ | ~0.78–0.79 |
| Top leaderboard (ensemble lớn) | ~0.805 |

Con số đáng nhớ: từ **0.75 lên 0.79 là nhờ dữ liệu, không nhờ mô hình.** Đổi thuật toán cho ~0.08 (0.67 → 0.75); thêm bảng phụ cho thêm ~0.04; tuning và ensemble cho phần còn lại ~0.015.

Thứ tự ưu tiên rút ra: **dữ liệu > feature engineering > thuật toán > tuning.**

## Kỹ thuật cần luyện trên bộ này

### 1. Aggregation nhiều tầng

Bảng depth 2 phải aggregate hai lần. Ví dụ với `bureau_balance`:

```python
# tầng 1: bureau_balance (SK_ID_BUREAU × tháng) → mỗi SK_ID_BUREAU một dòng
bb_agg = (bureau_balance
          .groupby('SK_ID_BUREAU')
          .agg(MONTHS_BALANCE_min=('MONTHS_BALANCE','min'),
               MONTHS_BALANCE_size=('MONTHS_BALANCE','size'),
               STATUS_worst=('STATUS','max')))

# tầng 2: bureau + bb_agg → mỗi SK_ID_CURR một dòng
bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
bureau_agg = bureau.groupby('SK_ID_CURR').agg(['min','max','mean','sum','count'])
```

Đặt tên cột có hệ thống ngay từ đầu, nếu không sẽ có 800 cột không truy được nguồn gốc.

### 2. Feature theo cửa sổ thời gian

`installments_payments` có `DAYS_INSTALMENT` và `DAYS_ENTRY_PAYMENT` → tính được:
```python
ins['DPD']    = (ins.DAYS_ENTRY_PAYMENT - ins.DAYS_INSTALMENT).clip(lower=0)
ins['DBD']    = (ins.DAYS_INSTALMENT - ins.DAYS_ENTRY_PAYMENT).clip(lower=0)
ins['PAYMENT_PERC'] = ins.AMT_PAYMENT / ins.AMT_INSTALMENT
ins['PAYMENT_DIFF'] = ins.AMT_INSTALMENT - ins.AMT_PAYMENT
```
Rồi aggregate `DPD` theo max / mean / và theo cửa sổ 3m / 6m / 12m. Nhóm feature DPD từ `installments_payments` là nhóm mạnh nhất ngoài `EXT_SOURCE_*`.

### 3. Bộ nhớ

2.68 GB CSV nở ra nhiều lần khi đọc bằng pandas mặc định. Kỹ thuật cần: downcast dtype (`float64`→`float32`, `int64`→`int32`), đọc theo chunk, dùng `category` cho biến phân loại, hoặc chuyển sang parquet/Polars.

## Rút ra cho dự án

**Copy được ngay:**
1. Sơ đồ quan hệ bảng — dùng làm template khi vẽ sơ đồ dữ liệu của dự án.
2. Phân biệt hai nhánh: lịch sử bên ngoài (bureau) vs lịch sử nội bộ (previous applications).
3. Mẫu aggregation hai tầng cho bảng depth 2.
4. Bộ feature DPD từ lịch sử trả góp.
5. Danh mục nhóm feature trong `application_train` — dùng làm checklist khảo sát dữ liệu.
6. Kỳ vọng thực tế: AUC 0.75–0.80 là tốt cho application scorecard.

**Chú ý:**
- `CODE_GENDER` dùng được trên Kaggle nhưng nhiều thị trường **cấm** dùng trong quyết định tín dụng.
- Bộ này **không có cột thời gian** ở `application_train` → không tập được out-of-time split. Đó chính là điều cuộc thi 2024 sửa lại bằng `date_decision` và `WEEK_NUM`.
- 47 cột nhà ở missing nhiều và trùng lặp — ví dụ tốt về việc "nhiều cột" không có nghĩa là "nhiều thông tin".

## Liên quan
- Notebook pipeline trên bộ này: [Report #1](01-nb-start-here-gentle-introduction.md)
- Notebook EDA trên bộ này: [Report #2](02-nb-complete-eda-feature-importance.md)
- Cuộc thi kế nhiệm: [Report #5](05-comp-home-credit-model-stability.md)
- Nguồn feature: [03-feature-den-tu-dau.md](../00-tong-quan/03-feature-den-tu-dau.md)

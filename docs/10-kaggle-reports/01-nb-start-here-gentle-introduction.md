# Report #1 — Start Here: A Gentle Introduction

**Link:** https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction
**Tác giả:** Will Koehrsen · đăng 2018 · 554,670 lượt xem · huy chương vàng · 561 comment
**Dữ liệu:** Home Credit Default Risk
**Điểm public LB tốt nhất:** 0.75262 (version 8)
**Runtime:** ~20 phút
**License:** Apache 2.0

> Ghi chú nguồn: lấy được **mục lục đầy đủ** + metadata từ trang Kaggle. Nội dung code cell không fetch được (trang render bằng JS). Phần mô tả từng mục dưới đây dựa trên mục lục cộng đặc điểm đã biết của bộ dữ liệu Home Credit Default Risk. Các mục có dấu ⚠️ là suy luận, cần đối chiếu khi mở notebook thật.

## Vì sao đọc notebook này đầu tiên

Đây là notebook nhập môn được tham chiếu nhiều nhất của cuộc thi. Giá trị không nằm ở điểm số (0.75 so với top ~0.80) mà ở chỗ nó đi **trọn vẹn một vòng** từ đọc dữ liệu tới submit, với giải thích cho từng quyết định. Đúng thứ cần cho Pre-Sprint 0.

## Mục lục (nguyên văn từ trang)

```
Introduction: Home Credit Default Risk Competition
Data
Metric: ROC AUC
Imports
Read in Data
Exploratory Data Analysis
  Examine the Distribution of the Target Column
  Examine Missing Values
  Column Types
  Encoding Categorical Variables
  Back to Exploratory Data Analysis
  Pairs Plot
Feature Engineering
  Polynomial Features
  Domain Knowledge Features
Baseline
  Logistic Regression Implementation
  Improved Model: Random Forest
  Model Interpretation: Feature Importances
Conclusions
Just for Fun: Light Gradient Boosting Machine
```

## Nội dung theo mục

### Metric: ROC AUC
Đặt metric **trước** khi động vào mô hình. Lý do: target imbalanced (~8% TARGET=1), accuracy vô nghĩa. Đây là thói quen tốt cần copy — chọn metric trước, không chọn sau khi thấy kết quả.

### Examine the Distribution of the Target Column
`value_counts()` trên TARGET. Kết luận: mất cân bằng, cần AUC + cân nhắc class weight.

### Examine Missing Values
Bảng số lượng + phần trăm missing theo cột. Home Credit Default Risk có nhiều cột thông tin nhà ở missing ~50–70%. ⚠️ Notebook giữ lại phần lớn thay vì bỏ, để mô hình tự xử lý.

### Column Types + Encoding Categorical Variables
Chiến lược lai, đây là mục đáng học nhất về mặt kỹ thuật:
- Biến category có **≤ 2 giá trị** → `LabelEncoder` (không sinh thứ tự giả vì chỉ có 2 mức).
- Biến category có **> 2 giá trị** → `pd.get_dummies()` one-hot.
- Sau one-hot, train và test lệch cột → phải `train.align(test, join='inner', axis=1)` rồi gắn lại TARGET.

Bước `align` là bug kinh điển của người mới: test thiếu một category nào đó → thiếu cột → model.predict lỗi hoặc lệch cột âm thầm.

### Back to Exploratory Data Analysis — anomaly
Đây là phần nổi tiếng nhất của notebook. Hai phát hiện:

1. **`DAYS_BIRTH`** âm (tính ngược từ ngày nộp hồ sơ). `DAYS_BIRTH / -365` = tuổi. Chia tuổi thành bin 5 năm rồi vẽ bad rate → khách trẻ rủi ro cao hơn rõ rệt, quan hệ đơn điệu.

2. **`DAYS_EMPLOYED` = 365243** — tương đương ~1000 năm đi làm, xuất hiện ở hơn 55,000 dòng. Đây là mã sentinel cho nhóm không đi làm. Cách xử lý đúng, copy nguyên:

```python
app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
```
Tạo cờ **trước**, thay NaN **sau**. Điều thú vị: nhóm anomaly này lại có bad rate **thấp hơn** trung bình → cờ đó là feature có ích, không phải rác.

### Pairs Plot
Ma trận scatter/KDE cho `EXT_SOURCE_1/2/3` + `DAYS_BIRTH`, tô màu theo TARGET. `EXT_SOURCE_*` là điểm chuẩn hóa từ nguồn dữ liệu ngoài và là 3 biến tương quan mạnh nhất với target trong toàn bộ `application_train`.

### Feature Engineering — Polynomial Features
Sinh tương tác và lũy thừa bậc 3 từ `EXT_SOURCE_1/2/3` + `DAYS_BIRTH` bằng `PolynomialFeatures(degree=3)`. ⚠️ Kết quả không cải thiện đáng kể — bài học: tương tác tự động không thay được hiểu biết nghiệp vụ, và sinh nhiều feature vô nghĩa làm chậm mọi thứ.

### Feature Engineering — Domain Knowledge Features
Bốn tỷ lệ, thứ đáng nhớ nhất của notebook:

```python
CREDIT_INCOME_PERCENT  = AMT_CREDIT   / AMT_INCOME_TOTAL
ANNUITY_INCOME_PERCENT = AMT_ANNUITY  / AMT_INCOME_TOTAL   # ≈ DTI
CREDIT_TERM            = AMT_ANNUITY  / AMT_CREDIT
DAYS_EMPLOYED_PERCENT  = DAYS_EMPLOYED / DAYS_BIRTH
```
Bốn dòng, dựa trên hiểu biết nghiệp vụ, hiệu quả hơn hàng trăm polynomial feature. Đây là ví dụ cụ thể cho nguyên tắc ở [03-feature-den-tu-dau.md](../00-tong-quan/03-feature-den-tu-dau.md#34-ratio-features--rẻ-và-mạnh).

### Baseline — Logistic Regression
Pipeline: `Imputer(strategy='median')` → `MinMaxScaler` → `LogisticRegression(C=0.0001)`.
`C` rất nhỏ = regularization rất mạnh, cần thiết vì có hàng trăm cột dummy.
⚠️ Public LB ≈ 0.67.

### Improved Model — Random Forest
`RandomForestClassifier(n_estimators=100, n_jobs=-1)`. ⚠️ LB ≈ 0.68. Chênh lệch nhỏ so với LR cho thấy phần lớn signal là tuyến tính trên bảng `application` — muốn hơn phải lấy dữ liệu từ các bảng phụ.

### Model Interpretation — Feature Importances
`feature_importances_` của RF. Top: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`, `EXT_SOURCE_1`, `DAYS_EMPLOYED`.
Notebook cảnh báo đúng: feature importance của tree bị thiên vị biến có cardinality cao, chỉ nên dùng để định hướng chứ không kết luận nhân quả.

### Just for Fun — LightGBM
LightGBM đưa điểm lên ~0.75 (điểm tốt nhất notebook: **0.75262**). Đây là bậc nhảy lớn nhất trong cả notebook và giải thích vì sao GBDT là mặc định cho tabular.

## Rút ra cho dự án

**Copy được ngay:**
1. Chọn metric trước khi mô hình hóa.
2. Quy tắc encode ≤2 giá trị → label, >2 → one-hot, rồi `align` train/test.
3. Mẫu xử lý anomaly: tạo cờ trước, thay NaN sau.
4. Bốn ratio feature nghiệp vụ.
5. Thứ tự leo thang mô hình: LR → RF → LightGBM, ghi lại điểm từng bậc để biết mỗi thay đổi đáng giá bao nhiêu.

**Không copy:**
- `PolynomialFeatures(degree=3)` — tốn tài nguyên, ít lợi.
- Chỉ dùng `application_train` — bỏ mất 6 bảng phụ, chính là chỗ chênh 0.75 → 0.80.
- Random train/test split — trong dự án thật phải split out-of-time.

**Số để neo kỳ vọng:** trên bài credit scoring thật, AUC 0.67 (LR đơn giản) → 0.75 (GBDT) → 0.80 (feature engineering kỹ). Không có phép màu nào đưa lên 0.95.

## Liên quan
- Bộ dữ liệu: [Report #7 — Home Credit Default Risk](07-comp-home-credit-default-risk.md)
- Notebook EDA sâu hơn trên cùng dữ liệu: [Report #2](02-nb-complete-eda-feature-importance.md)

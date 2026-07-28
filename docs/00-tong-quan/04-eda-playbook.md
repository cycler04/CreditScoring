# 4. EDA playbook cho dữ liệu tín dụng

Rút từ 4 notebook trong task.txt, sắp lại thành quy trình chạy được. Thứ tự có chủ ý — mỗi bước quyết định bước sau.

## Bước 0 — Khung dữ liệu

```python
df.shape, df.dtypes.value_counts()
df.head()
```
Ghi lại: số dòng, số cột, khóa chính, cột thời gian (dùng cho out-of-time split sau này).

## Bước 1 — Phân phối target

```python
df['TARGET'].value_counts(normalize=True)
```
Đây là bước đầu tiên của cả hai notebook Home Credit ("Examine the Distribution of the Target Column", "5.5 Data is balanced or imbalanced").

Cần biết:
- Bad rate bao nhiêu → quyết định metric (AUC, không phải accuracy) và có cần class_weight/scale_pos_weight không.
- Bad rate **theo tháng** — nếu nhảy bậc thì có thay đổi chính sách duyệt hoặc thay đổi định nghĩa label.

## Bước 2 — Missing values

```python
miss = (df.isna().mean() * 100).sort_values(ascending=False)
miss[miss > 0]
```

Quy tắc thực hành (notebook LendingClub dùng ngưỡng 51%):
- \> 50% missing → cân nhắc bỏ, **nhưng** tạo trước cờ `is_missing` và kiểm tra WoE của nhóm missing. Trong tín dụng, "không có dữ liệu" thường tự nó là tín hiệu rủi ro (khách mới, không có lịch sử).
- 5–50% → giữ, impute + cờ missing.
- < 5% → impute (median cho số, mode/`"Unknown"` cho category).

Home Credit Default Risk có 67 cột missing, nhiều cột housing (`COMMONAREA_*`, `NONLIVINGAPARTMENTS_*`) missing ~70%. Notebook gentle introduction chọn giữ hết và để mô hình xử lý — cả hai hướng đều chấp nhận được, miễn có lý do.

## Bước 3 — Anomaly / giá trị mã hóa

**Bước này rất dễ bị bỏ qua và rất tốn kém.** Ví dụ kinh điển từ Home Credit Default Risk: `DAYS_EMPLOYED` có giá trị `365243` (≈ 1000 năm đi làm) cho hơn 55,000 dòng — đó là mã sentinel cho "không đi làm / hưu trí". Xử lý đúng:

```python
df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == 365243
df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
```
Tạo cờ **rồi mới** thay bằng NaN — giữ lại thông tin, bỏ giá trị rác.

Cần soi: giá trị lặp bất thường (999, -1, 0, 9999999), số âm ở biến không thể âm, ngày tương lai, thu nhập cực đại (Home Credit Default Risk có `AMT_INCOME_TOTAL` = 117 triệu, ngoại lai một dòng).

```python
df.describe().T  # nhìn min/max trước, mean sau
```

## Bước 4 — Biến số: phân phối và quan hệ với target

Ba biểu đồ đủ dùng:
1. Histogram / KDE tách theo target (`sns.kdeplot(df.loc[df.TARGET==0, col])` vs `==1`) — thấy ngay biến có phân tách hay không.
2. Bad rate theo decile của biến — trục x là 10 nhóm bằng nhau, trục y là bad rate. Đây là biểu đồ hữu ích nhất cho credit scoring, cho thấy quan hệ có **đơn điệu** không.
3. Boxplot để soi outlier.

Ví dụ tìm được bằng cách này ở Home Credit Default Risk: khách trẻ có bad rate cao hơn rõ rệt — chia `DAYS_BIRTH/-365` thành các bin 5 năm, bad rate giảm đều từ nhóm 20–25 xuống nhóm 65–70.

## Bước 5 — Biến phân loại

```python
df[col].value_counts(dropna=False)
pd.crosstab(df[col], df['TARGET'], normalize='index')
```
Notebook "Complete EDA + Feature Importance" dành gần như toàn bộ mục 5.14 cho việc này: với từng biến (income source, family status, occupation, education, house type, organization type) vẽ **tỷ lệ repaid / not-repaid theo %**, không vẽ count. Lý do: count chỉ cho thấy nhóm nào đông, tỷ lệ mới cho thấy nhóm nào rủi ro.

Cần chú ý:
- Category tần suất rất thấp → gộp thành `other` (LendingClub gộp `medical`, `vacation`, `wedding`, `renewable_energy`, `educational`).
- Category có cấu trúc phân cấp trùng nhau → bỏ bớt (LendingClub bỏ `sub_grade` vì đã có `grade`).
- Cardinality cao → gộp theo vùng miền hoặc theo mức rủi ro, hoặc bỏ.

## Bước 6 — Tương quan

Hai loại, đừng lẫn:

**a) Tương quan với target** — để xếp hạng sơ bộ.
```python
df.corr()['TARGET'].sort_values()
```
Ở Home Credit Default Risk, kết quả nổi bật là `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1` (tương quan âm mạnh nhất) và `DAYS_BIRTH` (tương quan dương). Lưu ý Pearson chỉ bắt quan hệ tuyến tính; với target nhị phân nên ưu tiên **IV** (mục 6).

**b) Tương quan giữa các feature** — để bỏ trùng lặp.
```python
corr = df.corr().abs()
pairs = corr.stack()
pairs = pairs[(pairs > 0.8) & (pairs < 1.0)]
```
Quy tắc chọn bỏ (notebook LendingClub): trong mỗi cặp |corr| > 0.8, giữ biến có tương quan với target cao hơn. Hoặc tốt hơn: gộp thành ratio.

Đa cộng tuyến giết chết khả năng diễn giải của logistic regression (hệ số đổi dấu, sai với trực giác nghiệp vụ) — với GBDT thì ít hại hơn nhưng làm loãng feature importance.

## Bước 7 — Phân tích theo thời gian (bắt buộc, hay bị quên)

Vẽ theo tháng/tuần của `date_decision`:
- Số hồ sơ.
- Bad rate.
- Trung bình / tỷ lệ missing của vài feature chính.

Nếu ba đường này không ổn định → mô hình sẽ không ổn định. Đây chính là tiền đề của cuộc thi Model Stability và của PSI (xem [06-metrics-validation-monitoring.md](06-metrics-validation-monitoring.md)).

## Bước 8 — Segment analysis

Cắt bad rate theo các lát nghiệp vụ: kênh bán, sản phẩm, vùng, khách mới vs khách cũ, khoảng số tiền vay. Mục tiêu: phát hiện segment cần mô hình riêng, hoặc segment có bad rate bất thường (thường là dấu hiệu vấn đề quy trình, không phải vấn đề mô hình).

## Bước 9 — Sổ ghi phát hiện

Kết thúc EDA phải có một file liệt kê:
- Anomaly đã tìm thấy + cách xử lý.
- Cột nghi leakage.
- Cột bỏ + lý do.
- Feature đề xuất tạo.
- Câu hỏi cần hỏi bộ phận nghiệp vụ.

## Sai lầm thường gặp

| Sai lầm | Hậu quả |
|---|---|
| Chỉ nhìn count, không nhìn tỷ lệ bad | Bỏ sót biến mạnh ở nhóm nhỏ |
| Impute trước khi split train/test | Rò rỉ thống kê từ test |
| Dùng random split | Đánh giá lạc quan, không thấy drift |
| Bỏ cột missing nhiều mà không kiểm tra WoE | Vứt mất tín hiệu "khách không có lịch sử" |
| Tin AUC cao ngay lần đầu | Gần như luôn là leakage |
| EDA trên toàn bộ dữ liệu gồm cả test | Rò rỉ ngầm |

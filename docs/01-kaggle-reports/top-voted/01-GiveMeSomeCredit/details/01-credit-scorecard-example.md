# #1 — Credit ScoreCard example

## Hồ sơ nguồn

- Tác giả: `orange90`.
- Kaggle: [Credit ScoreCard example](https://www.kaggle.com/code/orange90/credit-scorecard-example).
- Snapshot: hạng 1, **336 vote**, ngày 2026-07-28.
- File đã đọc:
  [`credit-scorecard-example.ipynb`](../../../../notebooks/top-voted/GiveMeSomeCredit/01-credit-scorecard-example/credit-scorecard-example.ipynb).
- Quy mô: 99 cell (70 code, 29 markdown), 98,540 byte.
- SHA-256:
  `5744e4bed030a2c77bc12af4d6831efdd008c13c12d376fbacb595f1f29b9502`.
- Trạng thái artifact: không có output và không có cell đã execute.

## Tóm tắt

Đây là notebook có cấu trúc gần scorecard truyền thống nhất trong top 3:

```text
EDA
→ median imputation
→ manual/qcut binning
→ IV feature selection
→ WoE encoding
→ Logistic Regression
→ ROC-AUC/confusion matrix
→ quy đổi hệ số thành điểm
```

Giá trị lớn nhất là tính giải thích: code cho thấy trực tiếp quan hệ giữa bin,
WoE, hệ số logistic và điểm của từng bin. Tuy nhiên metric validation bị nhiễm
leakage vì toàn bộ preprocessing có học từ target được thực hiện trước split.
Scorecard cũng có mâu thuẫn nội bộ về tham số scale và chiều diễn giải điểm.

## Dữ liệu và EDA

Notebook đọc `cs-training.csv` và `cs-test.csv`, nhưng mô hình chỉ sử dụng train.
EDA đi qua missing, utilization, tuổi, ba biến delinquency, `DebtRatio`,
`MonthlyIncome`, số khoản vay và số người phụ thuộc.

Các quan sát hữu ích:

- Ba biến delinquency có cụm giá trị 96/98 bất thường.
- `DebtRatio` rất lớn thường đi cùng `MonthlyIncome` thiếu hoặc bằng 0/1.
- Utilization có đuôi cực dài.
- Notebook không vội xóa toàn bộ anomaly; đây là thái độ EDA đúng hơn việc coi mọi giá trị ngoài IQR là lỗi.

Điểm yếu: median được tính trên toàn bộ tập train Kaggle. Không có missing
indicator, không lưu fitted median thành artifact, và không có transform tương ứng cho `cs-test.csv`.

## Binning, IV và WoE

Notebook dùng bin thủ công cho `age`, dependents và delinquency; dùng `qcut(q=5)`
cho utilization, debt ratio, income và exposure. Sau đó:

1. tính IV cho mọi biến đã bin;
2. chọn năm biến có IV cao;
3. tính WoE cho các bin;
4. fit Logistic Regression trên năm cột WoE.

Năm biến được chọn:

- `RevolvingUtilizationOfUnsecuredLines`;
- `NumberOfTime30-59DaysPastDueNotWorse`;
- `age`;
- `NumberOfTimes90DaysLate`;
- `NumberOfTime60-89DaysPastDueNotWorse`.

### Vấn đề phương pháp

**Verified — leakage nghiêm trọng.** Median, `qcut`, IV, WoE và feature selection
đều được fit trên toàn bộ `df_train`; chỉ sau đó mới gọi `train_test_split`.
Target của validation vì thế đã ảnh hưởng trực tiếp tới encoding và lựa chọn biến.
Hướng dẫn scikit-learn yêu cầu split trước, và mọi transform có học thống kê phải
fit chỉ trên train/fold train
([Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)).

**Verified — xử lý zero count không ổn định.**

- `cal_IV` xóa mọi bin có `Bad == 0`, làm thay đổi mẫu số và có thể làm IV lệch.
- `cal_WOE` đổi bad count 0 thành 1 nhưng không smoothing đối xứng.
- Good count bằng 0 vẫn có thể tạo `log(.../0)`.

Pipeline local nên dùng smoothing có provenance (ví dụ cộng một hằng số cho cả
good và bad), minimum bin size và kiểm tra monotonicity.

## Split, mô hình và metric

Code dùng:

```python
train_test_split(..., test_size=0.2, random_state=42)
LogisticRegression(random_state=42)
```

Markdown lại nói validation chiếm 30%, trong khi code là 20%. Split không dùng
`stratify=y`, dù bad rate chỉ khoảng 6–7%. Tài liệu scikit-learn xác nhận
`stratify=None` là mặc định; muốn giữ tỷ lệ lớp phải truyền nhãn rõ ràng
([`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)).

Notebook tính accuracy, ROC curve/AUC và confusion matrix. Đây là lựa chọn metric
tốt hơn notebook #2 vì ROC-AUC khớp competition. Tuy nhiên snapshot không lưu
output, nên **không có AUC số nào có thể xác minh từ artifact**.

## Quy đổi scorecard

Điểm từng bin được tính:

```python
score = round(-coefficient * woe * B)
final_score = 650 + sum(bin_scores)
```

Notebook đặt `A = 650`, `B = 72.13`. Có ba mâu thuẫn:

1. Markdown nói PDO là 30 điểm khi odds gấp đôi, nhưng `B = 72.13` tương ứng
   `50 / ln(2)`, không phải `30 / ln(2)`.
2. Markdown mô tả 650 là benchmark tại odds 20:1, nhưng code cộng thẳng 650 như
   intercept/offset và không đưa model intercept vào score.
3. Phần kết luận mẫu vừa nói good sample có “lower scores”, vừa nói good trên 500
   và bad dưới 500.

Vì vậy **Unknown**: điểm sinh ra có đúng base odds/PDO mà tác giả dự định hay
không. AUC có thể vẫn xếp hạng tốt trong khi scale điểm sai.

Hàm deployment minh họa được ý tưởng map raw value → bin → điểm, nhưng có nhánh
`return null` trong Python; `null` không được định nghĩa. Mapping cũng phụ thuộc
chuỗi interval của pandas, dễ vỡ khi version/format thay đổi.

## Phần nên tái sử dụng

- Bảng artifact `(feature, bin, WoE, IV, coefficient, points)`.
- Tách rõ feature gốc, bin và điểm để có thể triển khai bằng SQL/rules engine.
- EDA riêng cho các mã delinquency 96/98 và đuôi utilization/debt ratio.
- Logistic Regression trên WoE để giữ khả năng giải thích.

## Phần phải viết lại

1. Split stratified 60/20/20 trước mọi bước học thống kê.
2. Fit imputer, bin edges, WoE, IV và feature selection chỉ trên train.
3. Freeze mapping rồi chỉ `transform` validation/test.
4. Dùng smoothing đối xứng, minimum bin size và kiểm tra unseen/missing bin.
5. Ghi rõ định nghĩa odds, chiều score, base score, base odds và PDO; đưa
   intercept vào phép quy đổi.
6. Đánh giá AUC/Gini/KS trên validation để chọn cấu hình; báo cáo cuối trên test
   khóa riêng.
7. Lưu config, seed, split indices, fitted bins và checksum input.

## Đánh giá cuối

- **Verified:** notebook minh họa được một baseline scorecard hoàn chỉnh.
- **Verified:** validation hiện tại bị leakage và không đủ làm benchmark.
- **Verified:** scale score có mâu thuẫn giữa prose và code.
- **Unknown:** AUC, accuracy và score distribution vì snapshot không lưu output.

Kết luận: dùng notebook như tài liệu kiến trúc scorecard, không dùng metric hoặc
copy implementation nguyên trạng.

## Nguồn

- Orange90, [Credit ScoreCard example](https://www.kaggle.com/code/orange90/credit-scorecard-example),
  Kaggle, truy cập 2026-07-29.
- scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html),
  truy cập 2026-07-29.
- scikit-learn, [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
  truy cập 2026-07-29.

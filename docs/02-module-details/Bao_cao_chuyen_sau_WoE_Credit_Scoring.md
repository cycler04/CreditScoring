# Báo cáo chuyên sâu về Weight of Evidence (WoE) trong Credit Scoring

## Mục lục

1. [Mục tiêu và phạm vi](#1-mục-tiêu-và-phạm-vi)  
2. [Bối cảnh Credit Scoring](#2-bối-cảnh-credit-scoring)  
3. [Khái niệm Good, Bad và biến mục tiêu](#3-khái-niệm-good-bad-và-biến-mục-tiêu)  
4. [Trực giác của Weight of Evidence](#4-trực-giác-của-weight-of-evidence)  
5. [Công thức WoE](#5-công-thức-woe)  
6. [Ví dụ tính WoE đầy đủ](#6-ví-dụ-tính-woe-đầy-đủ)  
7. [Tại sao cần binning trước khi tính WoE](#7-tại-sao-cần-binning-trước-khi-tính-woe)  
8. [Các phương pháp binning](#8-các-phương-pháp-binning)  
9. [Monotonic binning](#9-monotonic-binning)  
10. [Xử lý missing value, outlier và special value](#10-xử-lý-missing-value-outlier-và-special-value)  
11. [Smoothing khi một bin có Good hoặc Bad bằng 0](#11-smoothing-khi-một-bin-có-good-hoặc-bad-bằng-0)  
12. [Information Value](#12-information-value)  
13. [Ý nghĩa thống kê của WoE](#13-ý-nghĩa-thống-kê-của-woe)  
14. [WoE kết hợp Logistic Regression](#14-woe-kết-hợp-logistic-regression)  
15. [Chuyển Probability of Default thành Credit Score](#15-chuyển-probability-of-default-thành-credit-score)  
16. [Phân rã điểm theo từng đặc trưng](#16-phân-rã-điểm-theo-từng-đặc-trưng)  
17. [Ví dụ chấm điểm nhiều khách hàng](#17-ví-dụ-chấm-điểm-nhiều-khách-hàng)  
18. [Giải thích quyết định và adverse reason](#18-giải-thích-quyết-định-và-adverse-reason)  
19. [Quy trình huấn luyện đúng để tránh data leakage](#19-quy-trình-huấn-luyện-đúng-để-tránh-data-leakage)  
20. [Đánh giá mô hình WoE Scorecard](#20-đánh-giá-mô-hình-woe-scorecard)  
21. [Theo dõi độ ổn định sau triển khai](#21-theo-dõi-độ-ổn-định-sau-triển-khai)  
22. [So sánh WoE Scorecard với LightGBM](#22-so-sánh-woe-scorecard-với-lightgbm)  
23. [Cài đặt WoE bằng Python](#23-cài-đặt-woe-bằng-python)  
24. [Những lỗi thường gặp](#24-những-lỗi-thường-gặp)  
25. [Checklist triển khai cho dự án CreditScoring](#25-checklist-triển-khai-cho-dự-án-creditscoring)  
26. [Kết luận](#26-kết-luận)  

---

# 1. Mục tiêu và phạm vi

Báo cáo này trình bày chi tiết cách **Weight of Evidence (WoE)** được sử dụng trong
một hệ thống Credit Scoring truyền thống.

Mục tiêu không chỉ là đưa ra công thức, mà còn giải thích toàn bộ chuỗi xử lý:

```text
Dữ liệu lịch sử
      ↓
Xác định Good / Bad
      ↓
Chia bin
      ↓
Tính WoE
      ↓
Tính Information Value
      ↓
Huấn luyện Logistic Regression
      ↓
Ước lượng Probability of Default
      ↓
Chuyển thành Credit Score
      ↓
Giải thích điểm theo từng đặc trưng
      ↓
Áp dụng rule phê duyệt
```

Đầu ra cuối cùng không chỉ là một con số xác suất, mà có thể bao gồm:

- Probability of Default;
- Credit Score;
- mức rủi ro;
- quyết định theo business rule;
- các yếu tố làm tăng hoặc giảm điểm;
- bảng scorecard có thể kiểm toán;
- chỉ số theo dõi độ ổn định sau triển khai.

---

# 2. Bối cảnh Credit Scoring

Trong bài toán Credit Scoring, mô hình thường nhận thông tin có tại thời điểm khách
hàng đăng ký khoản vay và ước lượng khả năng khách hàng trở thành người vay xấu trong
một khoảng thời gian xác định.

Ví dụ đầu vào:

| Biến | Ý nghĩa |
|---|---|
| `age` | Tuổi khách hàng |
| `MonthlyIncome` | Thu nhập hàng tháng |
| `DebtRatio` | Tỷ lệ nghĩa vụ nợ |
| `RevolvingUtilization` | Tỷ lệ sử dụng hạn mức tín dụng |
| `NumberOfTimes30DaysLate` | Số lần trả chậm từ 30 ngày |
| `NumberOfTimes90DaysLate` | Số lần trả chậm từ 90 ngày |
| `NumberOfDependents` | Số người phụ thuộc |

Biến mục tiêu:

| Giá trị | Ý nghĩa |
|---:|---|
| `0` | Good: không vỡ nợ trong performance window |
| `1` | Bad: vỡ nợ hoặc quá hạn nghiêm trọng |

Mô hình học ánh xạ:

$$
X \longrightarrow P(Y=1\mid X)
$$

Trong đó:

- `X` là các đặc trưng của hồ sơ;
- `Y=1` là Bad;
- `P(Y=1|X)` là Probability of Default, viết tắt là `PD`.

---

# 3. Khái niệm Good, Bad và biến mục tiêu

Trước khi tính WoE, dự án phải xác định rõ thế nào là một khách hàng Good hoặc Bad.

Ví dụ:

```text
Observation window:
12 tháng dữ liệu trước ngày đăng ký

Performance window:
12 tháng sau ngày giải ngân

Bad:
Khách hàng quá hạn từ 90 ngày trở lên trong performance window

Good:
Khách hàng không thỏa định nghĩa Bad
```

Định nghĩa Good/Bad ảnh hưởng trực tiếp đến:

- số lượng Good và Bad;
- WoE của từng bin;
- IV của từng biến;
- hệ số Logistic Regression;
- xác suất vỡ nợ;
- Credit Score;
- cutoff phê duyệt.

Nếu định nghĩa Bad thay đổi, toàn bộ scorecard cần được đánh giá lại.

---

# 4. Trực giác của Weight of Evidence

Giả sử khách hàng có `DebtRatio = 73%`.

Giá trị `73%` chỉ mô tả tình trạng tài chính hiện tại. Nó chưa trực tiếp cho biết
mức độ rủi ro dựa trên dữ liệu lịch sử.

WoE đặt câu hỏi khác:

> Trong nhóm khách hàng có Debt Ratio tương tự, tỷ trọng Good so với tỷ trọng Bad
> trong toàn bộ dữ liệu là bao nhiêu?

Ví dụ:

```text
Nhóm Debt Ratio > 60%

Chiếm 5.6% tổng số khách hàng Good
Chiếm 46.0% tổng số khách hàng Bad
```

Nhóm này xuất hiện trong tập Bad nhiều hơn rất nhiều so với tập Good. Vì vậy đây là
bằng chứng mạnh cho rủi ro.

WoE biến đổi:

```text
DebtRatio = 73%
        ↓
Bin: DebtRatio > 60%
        ↓
WoE = -2.11
```

Có thể xem WoE như một **risk embedding một chiều**:

- WoE dương lớn: bằng chứng nghiêng về Good;
- WoE gần 0: bin xuất hiện với tỷ lệ gần giống nhau trong Good và Bad;
- WoE âm lớn: bằng chứng nghiêng về Bad.

> Lưu ý: Báo cáo này dùng quy ước `ln(%Good / %Bad)`. Một số thư viện dùng quy
> ước ngược lại `ln(%Bad / %Good)`. Hai quy ước đều hợp lệ nếu được sử dụng nhất quán.

---

# 5. Công thức WoE

Với bin thứ `i`, đặt:

- `Good_i`: số khách hàng Good trong bin `i`;
- `Bad_i`: số khách hàng Bad trong bin `i`;
- `TotalGood`: tổng số Good trong tập huấn luyện;
- `TotalBad`: tổng số Bad trong tập huấn luyện.

Phân phối Good trong bin:

$$
DistGood_i=\frac{Good_i}{TotalGood}
$$

Phân phối Bad trong bin:

$$
DistBad_i=\frac{Bad_i}{TotalBad}
$$

Weight of Evidence:

$$
WoE_i=\ln\left(\frac{DistGood_i}{DistBad_i}\right)
$$

Tương đương:

$$
WoE_i=
\ln\left(
\frac{Good_i/TotalGood}
{Bad_i/TotalBad}
\right)
$$

Có thể tách log:

$$
WoE_i=
\ln(Good_i)
-\ln(Bad_i)
-\ln(TotalGood)
+\ln(TotalBad)
$$

## Cách đọc dấu của WoE

| Trường hợp | Diễn giải |
|---|---|
| `WoE > 0` | Bin có tỷ trọng Good cao hơn tỷ trọng Bad |
| `WoE = 0` | Tỷ trọng Good và Bad bằng nhau |
| `WoE < 0` | Bin có tỷ trọng Bad cao hơn tỷ trọng Good |

Ví dụ:

$$
WoE=0
\Longleftrightarrow
DistGood=DistBad
$$

Nếu:

$$
DistGood=2\times DistBad
$$

thì:

$$
WoE=\ln(2)\approx0.693
$$

Nếu:

$$
DistBad=4\times DistGood
$$

thì:

$$
WoE=\ln\left(\frac{1}{4}\right)\approx-1.386
$$

---

# 6. Ví dụ tính WoE đầy đủ

Xét đặc trưng `DebtRatio`.

## 6.1 Chia bin

| Bin | Khoảng Debt Ratio |
|---|---|
| A | `0%–20%` |
| B | `20%–40%` |
| C | `40%–60%` |
| D | `>60%` |

## 6.2 Thống kê dữ liệu lịch sử

| Bin | Good | Bad | Tổng |
|---|---:|---:|---:|
| A | 400 | 20 | 420 |
| B | 300 | 80 | 380 |
| C | 150 | 170 | 320 |
| D | 50 | 230 | 280 |
| **Tổng** | **900** | **500** | **1,400** |

## 6.3 Tính phân phối Good và Bad

Bin A:

$$
DistGood_A=\frac{400}{900}=0.4444
$$

$$
DistBad_A=\frac{20}{500}=0.0400
$$

Bin B:

$$
DistGood_B=\frac{300}{900}=0.3333
$$

$$
DistBad_B=\frac{80}{500}=0.1600
$$

Bin C:

$$
DistGood_C=\frac{150}{900}=0.1667
$$

$$
DistBad_C=\frac{170}{500}=0.3400
$$

Bin D:

$$
DistGood_D=\frac{50}{900}=0.0556
$$

$$
DistBad_D=\frac{230}{500}=0.4600
$$

## 6.4 Tính WoE

Bin A:

$$
WoE_A=
\ln\left(\frac{0.4444}{0.0400}\right)
=\ln(11.1111)
\approx2.408
$$

Bin B:

$$
WoE_B=
\ln\left(\frac{0.3333}{0.1600}\right)
=\ln(2.0833)
\approx0.734
$$

Bin C:

$$
WoE_C=
\ln\left(\frac{0.1667}{0.3400}\right)
=\ln(0.4902)
\approx-0.713
$$

Bin D:

$$
WoE_D=
\ln\left(\frac{0.0556}{0.4600}\right)
=\ln(0.1208)
\approx-2.114
$$

## 6.5 Bảng WoE hoàn chỉnh

| Bin | Good | Bad | DistGood | DistBad | WoE |
|---|---:|---:|---:|---:|---:|
| A: `0%–20%` | 400 | 20 | 0.4444 | 0.0400 | 2.408 |
| B: `20%–40%` | 300 | 80 | 0.3333 | 0.1600 | 0.734 |
| C: `40%–60%` | 150 | 170 | 0.1667 | 0.3400 | -0.713 |
| D: `>60%` | 50 | 230 | 0.0556 | 0.4600 | -2.114 |

## 6.6 Biến đổi khách hàng mới

| Khách hàng | Debt Ratio | Bin | Giá trị đưa vào mô hình |
|---|---:|---|---:|
| A | 15% | A | 2.408 |
| B | 32% | B | 0.734 |
| C | 51% | C | -0.713 |
| D | 73% | D | -2.114 |

Logistic Regression không còn nhìn thấy trực tiếp `15%`, `32%`, `51%` hoặc `73%`.
Mô hình nhận mức bằng chứng rủi ro được suy ra từ dữ liệu lịch sử.

> Số liệu trong ví dụ được cố ý làm rõ sự khác biệt giữa các bin. Vì vậy IV ở ví dụ
> này rất lớn và không nên được xem là mức IV điển hình của dữ liệu thực tế.

---

# 7. Tại sao cần binning trước khi tính WoE

WoE không thường được tính cho từng giá trị số riêng lẻ. Các giá trị liên tục được gom
thành bin trước.

## 7.1 Giảm nhiễu

Nếu tính WoE riêng cho từng giá trị tuổi:

```text
age = 31
age = 32
age = 33
...
```

một số tuổi có ít dữ liệu, làm tỷ lệ Bad biến động mạnh.

Gom thành:

```text
18–25
26–35
36–50
51–65
>65
```

giúp ước lượng ổn định hơn.

## 7.2 Biểu diễn quan hệ phi tuyến

Logistic Regression trên giá trị gốc giả định:

$$
logit(PD)=\beta_0+\beta_1x
$$

Điều này có nghĩa mỗi một đơn vị tăng của `x` tạo ra cùng một thay đổi trong log-odds.

Trong thực tế, rủi ro có thể thay đổi theo ngưỡng:

```text
Utilization 0%–30%:
rủi ro gần như ổn định

Utilization 30%–70%:
rủi ro tăng từ từ

Utilization >90%:
rủi ro tăng mạnh
```

Binning cho phép mỗi vùng có một mức WoE riêng.

## 7.3 Hạn chế ảnh hưởng của outlier

Ví dụ `DebtRatio` có thể xuất hiện giá trị cực lớn do:

- mẫu số thu nhập rất nhỏ;
- thu nhập bị thiếu;
- lỗi nhập dữ liệu;
- cách định nghĩa biến đặc biệt.

Nếu dùng trực tiếp, một giá trị cực lớn có thể ảnh hưởng mạnh đến Logistic Regression.
Nếu đặt mọi giá trị trên ngưỡng vào một bin `>P99`, ảnh hưởng sẽ được kiểm soát.

## 7.4 Xử lý missing value có ý nghĩa

Missing không nhất thiết là ngẫu nhiên.

Ví dụ:

```text
MonthlyIncome bị thiếu
```

có thể xuất hiện nhiều hơn ở:

- khách hàng tự doanh;
- khách hàng không cung cấp chứng từ;
- nhóm hồ sơ có quy trình khác;
- nhóm khách hàng có rủi ro riêng.

Do đó missing thường được tạo thành một bin riêng và có WoE riêng.

---

# 8. Các phương pháp binning

## 8.1 Binning thủ công theo nghiệp vụ

Chuyên gia chọn ngưỡng dựa trên ý nghĩa nghiệp vụ.

Ví dụ tuổi:

| Bin | Khoảng |
|---|---|
| 1 | `<21` |
| 2 | `21–25` |
| 3 | `26–35` |
| 4 | `36–50` |
| 5 | `51–65` |
| 6 | `>65` |

### Ưu điểm

- dễ giải thích;
- bám sát chính sách;
- ổn định;
- dễ kiểm soát.

### Nhược điểm

- phụ thuộc chuyên gia;
- có thể bỏ lỡ pattern trong dữ liệu;
- khó mở rộng cho nhiều biến.

---

## 8.2 Equal-width binning

Chia miền giá trị thành các khoảng có cùng độ rộng.

Với giá trị từ `0` đến `100`, chia thành 5 bin:

```text
0–20
20–40
40–60
60–80
80–100
```

Độ rộng:

$$
Width=\frac{Max-Min}{K}
$$

Trong đó `K` là số bin.

### Nhược điểm chính

Nếu phân phối lệch, một số bin có rất nhiều quan sát trong khi một số bin gần như rỗng.

---

## 8.3 Equal-frequency hoặc quantile binning

Mỗi bin chứa số quan sát gần bằng nhau.

Ví dụ chia theo quintile:

| Bin | Phần trăm dữ liệu |
|---|---:|
| 1 | 0–20% |
| 2 | 20–40% |
| 3 | 40–60% |
| 4 | 60–80% |
| 5 | 80–100% |

### Ưu điểm

- tránh bin quá ít dữ liệu;
- phù hợp để tạo bin ban đầu;
- dễ tự động hóa.

### Nhược điểm

- các điểm cắt có thể khó giải thích;
- nhiều giá trị trùng nhau có thể làm số bin thực tế giảm;
- chưa tối ưu theo biến mục tiêu.

---

## 8.4 Decision-tree binning

Dùng cây quyết định một biến để tìm điểm chia làm tăng khả năng phân biệt Good/Bad.

Ý tưởng:

```text
DebtRatio <= 0.42?
├── Có
│   └── DebtRatio <= 0.18?
└── Không
    └── DebtRatio <= 0.73?
```

Các leaf trở thành bin.

### Ưu điểm

- sử dụng thông tin biến mục tiêu;
- tìm được ngưỡng có khả năng phân biệt;
- mô hình hóa phi tuyến.

### Rủi ro

- overfit nếu cây quá sâu;
- bin có ít quan sát;
- điểm cắt không ổn định giữa các sample;
- phải fit chỉ trên training set.

Các ràng buộc thường dùng:

- giới hạn `max_depth`;
- giới hạn số leaf;
- yêu cầu `min_samples_leaf`;
- yêu cầu tỷ lệ Bad tối thiểu;
- merge các bin gần nhau.

---

## 8.5 ChiMerge

ChiMerge bắt đầu với nhiều interval nhỏ, sau đó liên tục gộp hai bin liền kề có phân phối
Good/Bad giống nhau nhất.

Với hai bin liền kề, xây dựng bảng:

| | Good | Bad |
|---|---:|---:|
| Bin 1 | `G1` | `B1` |
| Bin 2 | `G2` | `B2` |

Thống kê chi-square:

$$
\chi^2=
\sum_{r}
\sum_{c}
\frac{(O_{rc}-E_{rc})^2}{E_{rc}}
$$

Trong đó:

- `O` là số quan sát thực tế;
- `E` là số kỳ vọng nếu hai bin có cùng phân phối Good/Bad.

Nếu `chi-square` nhỏ, hai bin có hành vi tương tự và có thể được gộp.

### Quy trình

1. Tạo nhiều pre-bin.
2. Tính `chi-square` cho từng cặp bin liền kề.
3. Gộp cặp có `chi-square` nhỏ nhất.
4. Tính lại.
5. Dừng khi đạt số bin mong muốn hoặc ngưỡng thống kê.

---

## 8.6 Optimal binning

Optimal binning tìm tập điểm cắt tối ưu theo một mục tiêu như:

- tối đa hóa IV;
- tối đa hóa Jensen-Shannon divergence;
- duy trì monotonicity;
- giới hạn số bin;
- đảm bảo kích thước tối thiểu;
- giữ missing hoặc special value riêng.

Đây là phương pháp mạnh nhưng phức tạp hơn và cần kiểm soát overfitting.

---

# 9. Monotonic binning

## 9.1 Khái niệm

Monotonic binning yêu cầu Bad Rate hoặc WoE thay đổi theo một hướng nhất quán khi giá trị
đặc trưng tăng.

Ví dụ hợp lý với `NumberOfTimes90DaysLate`:

| Số lần quá hạn | Bad Rate |
|---|---:|
| 0 | 2% |
| 1 | 8% |
| 2 | 20% |
| `>=3` | 45% |

Bad Rate tăng đơn điệu.

Tương ứng với quy ước `WoE = ln(Good/Bad)`, WoE thường giảm đơn điệu:

| Số lần quá hạn | WoE |
|---|---:|
| 0 | 1.20 |
| 1 | 0.30 |
| 2 | -0.80 |
| `>=3` | -1.90 |

## 9.2 Tại sao monotonicity quan trọng?

- score dễ giải thích;
- hành vi ổn định hơn;
- giảm pattern do nhiễu;
- tránh trường hợp tăng nợ nhưng điểm lại tăng bất thường;
- thuận lợi cho kiểm toán và phê duyệt mô hình.

## 9.3 Không phải biến nào cũng cần đơn điệu

Ví dụ tuổi có thể có dạng chữ U:

```text
Rất trẻ:
ít lịch sử tín dụng, rủi ro cao

Trung niên:
rủi ro thấp

Rất cao tuổi:
rủi ro có thể tăng lại
```

Ép đơn điệu cứng có thể làm mất thông tin thật.

Cần phân biệt:

- monotonicity hợp lý về nghiệp vụ;
- pattern phi đơn điệu có cơ sở;
- dao động do sample nhỏ;
- dao động do data quality.

## 9.4 Merge để đạt monotonicity

Ví dụ WoE ban đầu:

| Bin | WoE |
|---|---:|
| 0–20% | 1.10 |
| 20–30% | 0.70 |
| 30–40% | 0.82 |
| 40–60% | -0.40 |
| >60% | -1.50 |

`0.70 → 0.82` phá monotonicity nhẹ. Có thể gộp hai bin:

| Bin mới | WoE |
|---|---:|
| 0–20% | 1.10 |
| 20–40% | 0.75 |
| 40–60% | -0.40 |
| >60% | -1.50 |

---

# 10. Xử lý missing value, outlier và special value

## 10.1 Missing value

Không nên mặc định điền mean trước khi đánh giá missing.

Tạo bin:

```text
MonthlyIncome = Missing
```

Sau đó tính WoE riêng.

Ví dụ:

| Bin | Good | Bad | Bad Rate | WoE |
|---|---:|---:|---:|---:|
| Missing | 700 | 300 | 30% | -0.45 |
| `<=5 triệu` | 1,200 | 200 | 14% | 0.31 |
| `5–15 triệu` | 4,500 | 300 | 6% | 0.98 |
| `>15 triệu` | 2,800 | 100 | 3% | 1.43 |

Missing mang thông tin rủi ro và không nên bị trộn vào một giá trị trung bình giả tạo.

## 10.2 Outlier

Các lựa chọn:

- winsorize trước khi binning;
- tạo bin outlier riêng;
- đặt bin `>P99`;
- kiểm tra và sửa lỗi dữ liệu;
- loại bỏ chỉ khi có lý do rõ ràng.

Không nên loại outlier chỉ vì giá trị lớn. Trong Credit Scoring, giá trị cực đoan có thể
chính là tín hiệu rủi ro.

## 10.3 Special value

Một số hệ thống mã hóa:

```text
-999 = Missing
-1   = Không áp dụng
0    = Không có lịch sử
```

Các giá trị này phải được tách trước khi binning. Nếu không, thuật toán có thể hiểu
`-999` là một giá trị số rất nhỏ.

## 10.4 Unseen category khi inference

Ví dụ training chỉ có:

```text
Government
Private
Self-employed
```

Nhưng inference xuất hiện:

```text
Gig worker
```

Cần chính sách:

- map vào `Other`;
- dùng bin unseen có WoE trung lập;
- từ chối chấm điểm và yêu cầu cập nhật mapping;
- theo dõi tần suất category mới.

Không được để hệ thống tự gán ngẫu nhiên.

---

# 11. Smoothing khi một bin có Good hoặc Bad bằng 0

Nếu một bin không có Bad:

$$
DistBad_i=0
$$

khi đó:

$$
WoE_i=\ln\left(\frac{DistGood_i}{0}\right)
$$

WoE tiến tới dương vô cùng.

Nếu một bin không có Good, WoE tiến tới âm vô cùng.

Điều này gây:

- hệ số không ổn định;
- điểm số cực đoan;
- lỗi số học;
- overfitting.

## 11.1 Additive smoothing

Thêm hằng số nhỏ `alpha` vào mỗi count.

Với `K` bin:

$$
DistGood_i=
\frac{Good_i+\alpha}
{TotalGood+\alpha K}
$$

$$
DistBad_i=
\frac{Bad_i+\alpha}
{TotalBad+\alpha K}
$$

Sau đó:

$$
WoE_i=
\ln\left(
\frac{DistGood_i}{DistBad_i}
\right)
$$

Giá trị phổ biến của `alpha` trong thực hành có thể là `0.5` hoặc `1`, nhưng phải được
ghi nhận rõ trong pipeline.

## 11.2 Ví dụ

Giả sử:

```text
Good_i = 50
Bad_i = 0
TotalGood = 900
TotalBad = 500
K = 4
alpha = 0.5
```

Khi đó:

$$
DistGood_i=
\frac{50.5}{902}
\approx0.0560
$$

$$
DistBad_i=
\frac{0.5}{502}
\approx0.0010
$$

$$
WoE_i=
\ln\left(\frac{0.0560}{0.0010}\right)
\approx4.03
$$

Giá trị vẫn lớn, nhưng hữu hạn.

## 11.3 Cách tốt hơn smoothing trong nhiều trường hợp

Nếu bin có zero count vì quá ít dữ liệu, nên cân nhắc:

- merge với bin liền kề;
- tăng minimum bin size;
- giảm số bin;
- kiểm tra sample bias.

Smoothing không thay thế cho thiết kế bin hợp lý.

---

# 12. Information Value

## 12.1 Công thức

Đóng góp IV của bin `i`:

$$
IV_i=
(DistGood_i-DistBad_i)\times WoE_i
$$

IV của đặc trưng:

$$
IV=
\sum_i IV_i
$$

Thay công thức WoE:

$$
IV=
\sum_i
(DistGood_i-DistBad_i)
\ln\left(
\frac{DistGood_i}{DistBad_i}
\right)
$$

## 12.2 Tính IV cho ví dụ Debt Ratio

| Bin | DistGood | DistBad | WoE | IV contribution |
|---|---:|---:|---:|---:|
| A | 0.4444 | 0.0400 | 2.408 | 0.974 |
| B | 0.3333 | 0.1600 | 0.734 | 0.127 |
| C | 0.1667 | 0.3400 | -0.713 | 0.124 |
| D | 0.0556 | 0.4600 | -2.114 | 0.855 |
| **Tổng IV** | | | | **2.080** |

Ví dụ được cố ý tạo có mức tách Good/Bad rất mạnh, nên IV lớn bất thường.

## 12.3 Trực giác

IV lớn khi:

- `DistGood` và `DistBad` khác nhau đáng kể;
- WoE có độ lớn cao;
- nhiều bin phân biệt rõ Good và Bad.

IV nhỏ khi:

$$
DistGood_i\approx DistBad_i
$$

Khi đó:

$$
WoE_i\approx0
$$

và:

$$
IV_i\approx0
$$

## 12.4 Quy tắc diễn giải tham khảo

| IV | Mức độ phân biệt tham khảo |
|---:|---|
| `<0.02` | Gần như không có giá trị |
| `0.02–0.10` | Yếu |
| `0.10–0.30` | Trung bình |
| `0.30–0.50` | Mạnh |
| `>0.50` | Rất mạnh, cần kiểm tra leakage hoặc bias |

Đây không phải tiêu chuẩn tuyệt đối. IV phụ thuộc:

- kích thước mẫu;
- cách binning;
- định nghĩa target;
- tỷ lệ Bad;
- giai đoạn dữ liệu;
- phân khúc khách hàng.

## 12.5 Tại sao IV quá cao có thể đáng ngờ?

Ví dụ biến:

```text
NumberOfDaysAfterDefault
```

có thể phân biệt gần như hoàn hảo Good/Bad, nhưng biến này chỉ có sau khi default xảy ra.
Đó là data leakage.

Các biến khác cần kiểm tra:

- trạng thái thu hồi nợ;
- số ngày quá hạn sau giải ngân;
- cờ charge-off;
- kết quả xử lý khoản vay;
- thông tin được cập nhật sau thời điểm ra quyết định.

IV cao không tự động đồng nghĩa với biến tốt.

---

# 13. Ý nghĩa thống kê của WoE

## 13.1 WoE là log likelihood ratio theo bin

Với một khách hàng nằm trong bin `i`, xét:

$$
P(Bin_i\mid Good)
$$

và:

$$
P(Bin_i\mid Bad)
$$

Trong dữ liệu thực nghiệm:

$$
P(Bin_i\mid Good)\approx DistGood_i
$$

$$
P(Bin_i\mid Bad)\approx DistBad_i
$$

Do đó:

$$
WoE_i=
\ln\left(
\frac{P(Bin_i\mid Good)}
{P(Bin_i\mid Bad)}
\right)
$$

WoE là log của tỷ số khả năng quan sát bin đó dưới hai lớp Good và Bad.

## 13.2 Liên hệ với Bayes

Theo Bayes:

$$
\frac{P(Good\mid Bin_i)}
{P(Bad\mid Bin_i)}
=
\frac{P(Bin_i\mid Good)}
{P(Bin_i\mid Bad)}
\times
\frac{P(Good)}
{P(Bad)}
$$

Lấy log:

$$
\ln
\frac{P(Good\mid Bin_i)}
{P(Bad\mid Bin_i)}
=
WoE_i
+
\ln
\frac{P(Good)}
{P(Bad)}
$$

Điều này cho thấy:

- prior odds đến từ tỷ lệ Good/Bad toàn cục;
- WoE là phần bằng chứng do bin cụ thể cung cấp;
- WoE dương làm posterior nghiêng về Good;
- WoE âm làm posterior nghiêng về Bad.

## 13.3 Tại sao dùng log?

Tỷ số bằng chứng của nhiều đặc trưng được nhân với nhau dưới giả định đơn giản hóa.
Log chuyển phép nhân thành phép cộng:

$$
\ln(a\times b)=\ln(a)+\ln(b)
$$

Scorecard và Logistic Regression đều dựa trên tổng tuyến tính, nên log-ratio phù hợp về
mặt toán học.

## 13.4 Liên hệ với Naive Bayes

Nếu các đặc trưng độc lập có điều kiện theo lớp, log posterior odds có dạng:

$$
\ln
\frac{P(Good\mid X)}
{P(Bad\mid X)}
=
\ln
\frac{P(Good)}
{P(Bad)}
+
\sum_j
\ln
\frac{P(X_j\mid Good)}
{P(X_j\mid Bad)}
$$

Mỗi số hạng trong tổng có dạng WoE.

Trong scorecard thực tế, các đặc trưng không hoàn toàn độc lập. Logistic Regression học
hệ số cho từng biến để điều chỉnh mức đóng góp thay vì mặc định mọi WoE có trọng số bằng 1.

---

# 14. WoE kết hợp Logistic Regression

## 14.1 Biến đổi dữ liệu

Với khách hàng `n`, mỗi đặc trưng `j` được map vào WoE:

$$
x_{nj}
\longrightarrow
WoE_{nj}
$$

Vector đầu vào:

$$
\mathbf{w}_n=
[WoE_{n1},WoE_{n2},\ldots,WoE_{np}]
$$

## 14.2 Mô hình Logistic Regression

Nếu target `Y=1` là Bad:

$$
z_n=
\beta_0+
\sum_{j=1}^{p}
\beta_j WoE_{nj}
$$

Probability of Default:

$$
PD_n=
P(Y=1\mid\mathbf{w}_n)
=
\frac{1}{1+\exp(-z_n)}
$$

Log-odds của Bad:

$$
\ln\left(
\frac{PD_n}{1-PD_n}
\right)
=
z_n
$$

## 14.3 Dấu hệ số

Báo cáo dùng:

$$
WoE=\ln\left(\frac{Good}{Bad}\right)
$$

Bin rủi ro thường có WoE âm. Để WoE âm làm tăng log-odds của Bad, hệ số thường có xu
hướng âm:

$$
\beta_j<0
$$

Ví dụ:

$$
WoE=-2.0
$$

$$
\beta=-0.8
$$

Đóng góp:

$$
\beta\times WoE=(-0.8)\times(-2.0)=1.6
$$

`z` tăng, nên `PD` tăng.

Nếu thư viện dùng quy ước:

$$
WoE=\ln\left(\frac{Bad}{Good}\right)
$$

thì hệ số thường có dấu ngược lại.

## 14.4 Ví dụ mô hình

Giả sử mô hình:

$$
z=
-1.20
-0.35WoE_{Age}
-0.75WoE_{Utilization}
-0.90WoE_{LatePayment}
-0.45WoE_{DebtRatio}
$$

Khách hàng có:

| Biến | WoE |
|---|---:|
| Age | 0.50 |
| Utilization | -1.20 |
| Late Payment | -1.50 |
| Debt Ratio | -0.70 |

Khi đó:

$$
z=
-1.20
-0.35(0.50)
-0.75(-1.20)
-0.90(-1.50)
-0.45(-0.70)
$$

$$
z=
-1.20
-0.175
+0.900
+1.350
+0.315
$$

$$
z=1.190
$$

Probability of Default:

$$
PD=
\frac{1}{1+\exp(-1.190)}
\approx0.767
$$

Khách hàng có PD khoảng `76.7%` trong ví dụ minh họa này.

---

# 15. Chuyển Probability of Default thành Credit Score

## 15.1 Good-to-Bad odds

Từ PD, odds Good so với Bad:

$$
Odds=
\frac{1-PD}{PD}
$$

Ví dụ:

$$
PD=0.02
$$

$$
Odds=
\frac{0.98}{0.02}
=49
$$

Có thể đọc là khoảng `49 Good : 1 Bad`.

## 15.2 Công thức score tổng quát

Credit Score thường tuyến tính theo log-odds:

$$
Score=
Offset+
Factor\times\ln(Odds)
$$

Vì:

$$
Odds=
\frac{1-PD}{PD}
$$

nên:

$$
Score=
Offset+
Factor
\ln\left(
\frac{1-PD}{PD}
\right)
$$

PD thấp tạo odds cao và score cao.

## 15.3 Points to Double Odds

`PDO` là số điểm tăng thêm khi Good-to-Bad odds tăng gấp đôi.

Yêu cầu:

$$
Score(2\times Odds)-Score(Odds)=PDO
$$

Từ công thức score:

$$
Factor\ln(2)=PDO
$$

Suy ra:

$$
Factor=
\frac{PDO}{\ln(2)}
$$

## 15.4 Base score và base odds

Giả sử:

- Base Score = `600`;
- Base Odds = `50:1`;
- PDO = `20`.

Factor:

$$
Factor=
\frac{20}{\ln(2)}
\approx28.854
$$

Offset được tìm từ:

$$
600=
Offset+
28.854\ln(50)
$$

Suy ra:

$$
Offset=
600-
28.854\ln(50)
\approx487.11
$$

Công thức score:

$$
Score=
487.11+
28.854
\ln\left(
\frac{1-PD}{PD}
\right)
$$

## 15.5 Ví dụ PD sang score

### PD bằng 2%

$$
Odds=
\frac{0.98}{0.02}
=49
$$

$$
Score=
487.11+
28.854\ln(49)
\approx599.4
$$

### PD bằng 10%

$$
Odds=
\frac{0.90}{0.10}
=9
$$

$$
Score=
487.11+
28.854\ln(9)
\approx550.5
$$

### PD bằng 30%

$$
Odds=
\frac{0.70}{0.30}
\approx2.333
$$

$$
Score=
487.11+
28.854\ln(2.333)
\approx511.6
$$

## 15.6 Score range 300–850

Khoảng `300–850` thường là lựa chọn trình bày hoặc chính sách hệ thống, không phải đầu
ra được Logistic Regression tự học.

Có thể:

- giữ score liên tục;
- làm tròn;
- clip vào `[300, 850]`;
- scale theo range khác;
- dùng base score và PDO riêng của tổ chức.

Nếu clip score, cần giữ cả raw score để phân tích vì clipping làm mất một phần thứ tự ở
hai đầu phân phối.

---

# 16. Phân rã điểm theo từng đặc trưng

Từ Logistic Regression:

$$
z=
\beta_0+
\sum_j\beta_jWoE_j
$$

Vì:

$$
\ln(Odds)=-z
$$

nên:

$$
Score=
Offset-Factor\times z
$$

Thay `z`:

$$
Score=
Offset
-Factor\beta_0
-\sum_jFactor\beta_jWoE_j
$$

Đặt base points:

$$
BasePoints=
Offset-Factor\beta_0
$$

Điểm đóng góp của biến `j`:

$$
Points_j=
-Factor\beta_jWoE_j
$$

Do đó:

$$
Score=
BasePoints+
\sum_jPoints_j
$$

Đây là cơ sở toán học cho khả năng giải thích scorecard.

## 16.1 Ví dụ

Giả sử:

$$
Factor=28.854
$$

và:

$$
\beta_{DebtRatio}=-0.45
$$

Khách hàng thuộc bin có:

$$
WoE_{DebtRatio}=-2.114
$$

Điểm đóng góp:

$$
Points_{DebtRatio}
=
-28.854
\times(-0.45)
\times(-2.114)
$$

$$
Points_{DebtRatio}
\approx-27.4
$$

Debt Ratio cao làm giảm khoảng `27.4` điểm so với phần base.

Nếu khách hàng thuộc bin an toàn:

$$
WoE_{DebtRatio}=2.408
$$

thì:

$$
Points_{DebtRatio}
=
-28.854
\times(-0.45)
\times2.408
$$

$$
Points_{DebtRatio}
\approx31.3
$$

Debt Ratio thấp làm tăng khoảng `31.3` điểm.

## 16.2 Điểm tương đối so với bin chuẩn

Trong thực tế, giải thích thường dùng chênh lệch so với:

- bin tốt nhất;
- bin trung tính;
- bin tham chiếu;
- điểm tối đa có thể nhận ở biến đó.

Nếu điểm thực tế của khách hàng là `-27` và điểm tốt nhất của biến là `+31`, phần điểm
bị mất là:

$$
LostPoints=31-(-27)=58
$$

Có thể báo:

```text
Debt Ratio cao làm khách hàng mất 58 điểm so với mức tốt nhất của đặc trưng này.
```

---

# 17. Ví dụ chấm điểm nhiều khách hàng

Giả sử scorecard gồm bốn biến:

- Age;
- Credit Utilization;
- Late Payment;
- Debt Ratio.

Base points là `520`.

## 17.1 Bảng điểm rút gọn

### Age

| Bin | Points |
|---|---:|
| `<25` | -15 |
| `25–35` | 5 |
| `36–50` | 18 |
| `>50` | 25 |

### Credit Utilization

| Bin | Points |
|---|---:|
| `0%–20%` | 45 |
| `20%–50%` | 20 |
| `50%–80%` | -20 |
| `>80%` | -65 |

### Late Payment

| Bin | Points |
|---|---:|
| `0` | 55 |
| `1` | 10 |
| `2` | -35 |
| `>=3` | -85 |

### Debt Ratio

| Bin | Points |
|---|---:|
| `0%–20%` | 35 |
| `20%–40%` | 15 |
| `40%–60%` | -20 |
| `>60%` | -55 |

> Bảng này chỉ minh họa cách sử dụng scorecard. Điểm thực tế phải được suy ra từ
> hệ số và WoE của mô hình đã huấn luyện.

## 17.2 Dữ liệu khách hàng

| Khách hàng | Age | Utilization | Late Payment | Debt Ratio |
|---|---:|---:|---:|---:|
| An | 42 | 18% | 0 | 25% |
| Bình | 29 | 72% | 2 | 58% |
| Chi | 55 | 35% | 0 | 15% |
| Dũng | 23 | 92% | 4 | 75% |

## 17.3 Tính điểm An

| Thành phần | Điểm |
|---|---:|
| Base | 520 |
| Age 36–50 | +18 |
| Utilization 0–20% | +45 |
| Late Payment 0 | +55 |
| Debt Ratio 20–40% | +15 |
| **Tổng** | **653** |

## 17.4 Tính điểm Bình

| Thành phần | Điểm |
|---|---:|
| Base | 520 |
| Age 25–35 | +5 |
| Utilization 50–80% | -20 |
| Late Payment 2 | -35 |
| Debt Ratio 40–60% | -20 |
| **Tổng** | **450** |

## 17.5 Tính điểm Chi

| Thành phần | Điểm |
|---|---:|
| Base | 520 |
| Age >50 | +25 |
| Utilization 20–50% | +20 |
| Late Payment 0 | +55 |
| Debt Ratio 0–20% | +35 |
| **Tổng** | **655** |

## 17.6 Tính điểm Dũng

| Thành phần | Điểm |
|---|---:|
| Base | 520 |
| Age <25 | -15 |
| Utilization >80% | -65 |
| Late Payment >=3 | -85 |
| Debt Ratio >60% | -55 |
| **Tổng** | **300** |

## 17.7 Rule quyết định minh họa

| Score | Quyết định |
|---:|---|
| `>=620` | Approve |
| `520–619` | Manual review |
| `<520` | Reject |

Kết quả:

| Khách hàng | Score | Quyết định |
|---|---:|---|
| An | 653 | Approve |
| Bình | 450 | Reject |
| Chi | 655 | Approve |
| Dũng | 300 | Reject |

Rule trên không được model học. Đây là một lớp business policy được áp dụng sau score.

---

# 18. Giải thích quyết định và adverse reason

## 18.1 Giải thích cho khách hàng Bình

Score của Bình là `450`.

Các yếu tố bất lợi:

| Yếu tố | Giá trị | Điểm hiện tại | Điểm tốt nhất | Điểm bị mất |
|---|---:|---:|---:|---:|
| Late Payment | 2 | -35 | +55 | 90 |
| Utilization | 72% | -20 | +45 | 65 |
| Debt Ratio | 58% | -20 | +35 | 55 |
| Age | 29 | +5 | +25 | 20 |

Có thể xếp hạng adverse reason:

1. Có nhiều lần trả chậm.
2. Tỷ lệ sử dụng hạn mức cao.
3. Tỷ lệ nợ cao.
4. Nhóm tuổi có lịch sử rủi ro cao hơn nhóm tham chiếu.

## 18.2 Explainability của scorecard là exact

Với scorecard tuyến tính:

$$
Score=
BasePoints+
Points_{Age}+
Points_{Utilization}+
Points_{LatePayment}+
Points_{DebtRatio}
$$

Tổng các đóng góp bằng chính xác score cuối.

Điều này khác với giải thích hậu nghiệm của mô hình phức tạp. Ví dụ SHAP phân rã dự đoán
của mô hình cây, nhưng scorecard đã có cấu trúc cộng điểm ngay từ thiết kế.

## 18.3 Không nên diễn giải thành quan hệ nhân quả

WoE cho biết quan hệ thống kê lịch sử, không chứng minh nguyên nhân.

Ví dụ:

```text
Nhóm có MonthlyIncome missing có Bad Rate cao
```

không đồng nghĩa việc điền thu nhập sẽ làm giảm rủi ro thật.

Giải thích đúng:

> Trong dữ liệu huấn luyện, hồ sơ thiếu thông tin thu nhập xuất hiện nhiều hơn trong
> nhóm Bad.

Giải thích sai:

> Thiếu thu nhập gây ra vỡ nợ.

---

# 19. Quy trình huấn luyện đúng để tránh data leakage

## 19.1 Sai lầm phổ biến

Quy trình sai:

```text
Toàn bộ dữ liệu
      ↓
Tìm bin và tính WoE
      ↓
Chia train / validation / test
```

WoE đã sử dụng label của validation và test. Đây là target leakage.

## 19.2 Quy trình đúng

```text
Dữ liệu gốc
      ↓
Chia train / validation / test
      ↓
Fit binning trên train
      ↓
Fit WoE mapping trên train
      ↓
Transform train bằng mapping train
      ↓
Transform validation bằng mapping train
      ↓
Transform test bằng mapping train
      ↓
Fit Logistic Regression trên train
      ↓
Chọn hyperparameter bằng validation
      ↓
Báo cáo cuối cùng trên test
```

## 19.3 Những đối tượng phải được lưu

Artifact của pipeline:

- danh sách biến;
- kiểu dữ liệu;
- quy tắc missing/special;
- điểm cắt bin;
- category mapping;
- WoE của từng bin;
- smoothing constant;
- hệ số Logistic Regression;
- intercept;
- Factor;
- Offset;
- Base Score;
- Base Odds;
- PDO;
- score cutoff;
- phiên bản dữ liệu và code.

## 19.4 Cross-validation

Nếu dùng cross-validation để đánh giá, binning và WoE phải được fit lại bên trong từng fold.

Sai:

```text
Fit WoE một lần trên toàn bộ train
      ↓
Cross-validation Logistic Regression
```

Đúng:

```text
Fold 1:
fit binning + WoE trên train fold
transform validation fold
fit Logistic Regression

Fold 2:
fit lại binning + WoE
...
```

## 19.5 Out-of-time validation

Nếu dữ liệu có thời gian, nên dùng:

```text
Train:
tháng 1–12 năm trước

Validation:
tháng 1–3 năm sau

Test:
tháng 4–6 năm sau
```

Out-of-time validation phản ánh tốt hơn khả năng scorecard hoạt động trên quần thể tương lai.

Nếu dataset không có timestamp, chỉ có thể dùng random hoặc stratified split và phải ghi
rõ hạn chế này.

---

# 20. Đánh giá mô hình WoE Scorecard

## 20.1 AUC

AUC đo khả năng xếp hạng một Bad cao rủi ro hơn một Good.

$$
AUC=
P(ScoreRisk_{Bad}>ScoreRisk_{Good})
$$

Nếu dùng Credit Score theo hướng điểm cao là tốt, khi tính ROC cần đảm bảo hướng score phù hợp,
hoặc dùng `PD` làm risk score.

## 20.2 Gini

$$
Gini=2\times AUC-1
$$

Ví dụ:

$$
AUC=0.75
$$

$$
Gini=2(0.75)-1=0.50
$$

## 20.3 KS

Tại mỗi threshold, tính:

$$
KS(t)=TPR(t)-FPR(t)
$$

KS tối đa:

$$
KS=\max_t|TPR(t)-FPR(t)|
$$

KS đo khoảng cách lớn nhất giữa phân phối tích lũy của Good và Bad.

## 20.4 Calibration

Nếu mô hình dự đoán `PD=10%` cho một nhóm, lý tưởng khoảng `10%` khách hàng trong nhóm
đó thực sự trở thành Bad.

Có thể kiểm tra theo decile:

| PD trung bình | Bad Rate thực tế |
|---:|---:|
| 2.1% | 2.4% |
| 4.8% | 5.0% |
| 8.2% | 8.9% |
| 15.3% | 14.7% |
| 30.1% | 28.8% |

Khả năng xếp hạng tốt không đảm bảo calibration tốt.

## 20.5 Lift và bad capture rate

Ví dụ top 10% rủi ro nhất chứa 35% toàn bộ Bad:

$$
BadCapture_{10\%}=35\%
$$

Lift:

$$
Lift_{10\%}=
\frac{35\%}{10\%}
=3.5
$$

Nhóm 10% rủi ro nhất chứa Bad nhiều gấp 3.5 lần chọn ngẫu nhiên.

## 20.6 Approval analysis

Với mỗi cutoff:

| Cutoff | Approval Rate | Bad Rate trong nhóm approve |
|---:|---:|---:|
| 500 | 85% | 9.0% |
| 550 | 72% | 5.8% |
| 600 | 55% | 3.1% |
| 650 | 32% | 1.5% |

Đây là phân tích policy. Cutoff không nên được gọi là đầu ra tự học của model nếu không có
hàm mục tiêu kinh doanh và dữ liệu chi phí/lợi nhuận.

---

# 21. Theo dõi độ ổn định sau triển khai

## 21.1 Population Stability Index

PSI so sánh phân phối của một biến hoặc score giữa tập tham chiếu và dữ liệu mới.

Với bin `i`:

- `Expected_i`: tỷ lệ trong dữ liệu tham chiếu;
- `Actual_i`: tỷ lệ trong dữ liệu hiện tại.

$$
PSI=
\sum_i
(Actual_i-Expected_i)
\ln\left(
\frac{Actual_i}{Expected_i}
\right)
$$

## 21.2 Ví dụ

| Score band | Expected | Actual |
|---|---:|---:|
| 300–500 | 10% | 18% |
| 500–600 | 25% | 30% |
| 600–700 | 40% | 35% |
| 700–850 | 25% | 17% |

Nếu dữ liệu mới dịch chuyển về score thấp, ngân hàng cần điều tra:

- thay đổi kênh acquisition;
- kinh tế suy giảm;
- thay đổi chính sách;
- lỗi dữ liệu;
- thay đổi định nghĩa biến;
- missing rate tăng;
- nhóm khách hàng mới.

## 21.3 Theo dõi WoE theo thời gian

WoE của một bin có thể thay đổi.

WoE lúc train:

$$
WoE_{train,i}=
\ln\left(
\frac{DistGood_{train,i}}
{DistBad_{train,i}}
\right)
$$

WoE ở thời gian mới:

$$
WoE_{new,i}=
\ln\left(
\frac{DistGood_{new,i}}
{DistBad_{new,i}}
\right)
$$

Chênh lệch lớn cho thấy quan hệ giữa đặc trưng và rủi ro đã thay đổi.

Tuy nhiên, chỉ có thể tính WoE mới sau khi performance window hoàn tất và label đã mature.

## 21.4 Các chỉ số nên theo dõi

- score PSI;
- feature PSI;
- missing rate;
- out-of-range rate;
- unseen category rate;
- approval rate;
- observed Bad Rate;
- AUC/Gini/KS theo thời gian;
- calibration;
- WoE drift;
- coefficient stability;
- số lượng hồ sơ theo bin.

---

# 22. So sánh WoE Scorecard với LightGBM

| Tiêu chí | WoE + Logistic Regression | LightGBM |
|---|---|---|
| Quan hệ phi tuyến | Qua binning | Học trực tiếp |
| Tương tác đặc trưng | Hạn chế | Mạnh |
| Khả năng giải thích | Rất trực tiếp | Cần SHAP hoặc công cụ khác |
| Score contribution | Chính xác theo công thức | Giải thích hậu nghiệm |
| Xử lý missing | Bin riêng | Xử lý split nội bộ |
| Binning thủ công | Cần | Không bắt buộc |
| Hiệu năng dự báo | Có thể thấp hơn | Thường mạnh hơn |
| Tính ổn định | Dễ kiểm soát | Cần kiểm soát kỹ |
| Kiểm toán | Đơn giản | Phức tạp hơn |
| Triển khai rule-based | Dễ | Khó hơn |
| Phù hợp scorecard truyền thống | Rất phù hợp | Không trực tiếp |

## 22.1 Khi nên dùng WoE Scorecard

- cần mô hình dễ giải thích;
- cần bảng điểm rõ ràng;
- cần triển khai trong hệ thống rule engine;
- dữ liệu dạng tabular;
- yêu cầu kiểm toán cao;
- sample không quá lớn;
- ưu tiên ổn định hơn vài điểm AUC.

## 22.2 Khi nên dùng LightGBM

- ưu tiên predictive performance;
- nhiều tương tác phi tuyến;
- dữ liệu lớn;
- có hạ tầng explainability;
- có quy trình model governance phù hợp;
- scorecard truyền thống không phải yêu cầu bắt buộc.

## 22.3 Champion–challenger

Một thiết kế thực tế:

```text
Champion:
WoE + Logistic Regression

Challenger:
LightGBM
```

So sánh:

- AUC/Gini/KS;
- calibration;
- stability;
- fairness;
- explainability;
- vận hành;
- lợi ích kinh doanh.

Không nên chọn model chỉ dựa trên AUC.

---

# 23. Cài đặt WoE bằng Python

Phần này minh họa một phiên bản tối giản bằng `pandas` và `scikit-learn`.

## 23.1 Tính bảng WoE

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WoeTableResult:
    table: pd.DataFrame
    information_value: float


def build_woe_table(
    binned_feature: pd.Series,
    target: pd.Series,
    *,
    bad_label: int = 1,
    smoothing: float = 0.5,
) -> WoeTableResult:
    if len(binned_feature) != len(target):
        raise ValueError("Feature và target phải có cùng số dòng.")

    if smoothing <= 0:
        raise ValueError("smoothing phải lớn hơn 0.")

    frame = pd.DataFrame(
        {
            "bin": binned_feature.astype("object").fillna("__MISSING__"),
            "target": target,
        }
    )

    valid_labels = set(frame["target"].dropna().unique())
    if not valid_labels.issubset({0, 1}):
        raise ValueError("Target phải là biến nhị phân 0/1.")

    frame["is_bad"] = (frame["target"] == bad_label).astype(int)
    frame["is_good"] = 1 - frame["is_bad"]

    grouped = (
        frame.groupby("bin", dropna=False, observed=False)
        .agg(
            count=("target", "size"),
            good=("is_good", "sum"),
            bad=("is_bad", "sum"),
        )
        .reset_index()
    )

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()
    number_of_bins = len(grouped)

    if total_good == 0 or total_bad == 0:
        raise ValueError("Dữ liệu phải chứa cả Good và Bad.")

    grouped["dist_good"] = (
        grouped["good"] + smoothing
    ) / (
        total_good + smoothing * number_of_bins
    )

    grouped["dist_bad"] = (
        grouped["bad"] + smoothing
    ) / (
        total_bad + smoothing * number_of_bins
    )

    grouped["woe"] = np.log(
        grouped["dist_good"] / grouped["dist_bad"]
    )

    grouped["iv_component"] = (
        grouped["dist_good"] - grouped["dist_bad"]
    ) * grouped["woe"]

    grouped["bad_rate"] = grouped["bad"] / grouped["count"]

    information_value = float(grouped["iv_component"].sum())

    return WoeTableResult(
        table=grouped.sort_values("bin").reset_index(drop=True),
        information_value=information_value,
    )
```

## 23.2 Tạo bin cho Debt Ratio

```python
import numpy as np
import pandas as pd


debt_bins = [-np.inf, 0.20, 0.40, 0.60, np.inf]
debt_labels = [
    "A_0_20",
    "B_20_40",
    "C_40_60",
    "D_over_60",
]

train_debt_bin = pd.cut(
    train_df["DebtRatio"],
    bins=debt_bins,
    labels=debt_labels,
    right=False,
)

result = build_woe_table(
    binned_feature=train_debt_bin,
    target=train_df["default"],
    bad_label=1,
    smoothing=0.5,
)

print(result.table)
print("IV:", result.information_value)
```

## 23.3 Tạo mapping

```python
woe_mapping = dict(
    zip(
        result.table["bin"].astype(str),
        result.table["woe"],
    )
)
```

## 23.4 Transform validation và test

```python
def transform_debt_ratio_to_woe(
    debt_ratio: pd.Series,
    *,
    bins: list[float],
    labels: list[str],
    mapping: dict[str, float],
    missing_woe: float,
) -> pd.Series:
    binned = pd.cut(
        debt_ratio,
        bins=bins,
        labels=labels,
        right=False,
    )

    binned_as_text = (
        binned.astype("object")
        .where(binned.notna(), "__MISSING__")
        .astype(str)
    )

    transformed = binned_as_text.map(mapping)

    return transformed.fillna(missing_woe).astype(float)
```

Quan trọng: `bins`, `labels` và `mapping` được fit từ training set. Không tính lại trên
validation hoặc test.

## 23.5 Huấn luyện Logistic Regression

```python
from sklearn.linear_model import LogisticRegression


feature_names = [
    "age_woe",
    "utilization_woe",
    "late_payment_woe",
    "debt_ratio_woe",
]

X_train = train_woe[feature_names]
y_train = train_df["default"]

model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=2_000,
)

model.fit(X_train, y_train)

pd_validation = model.predict_proba(
    validation_woe[feature_names]
)[:, 1]
```

## 23.6 Chuyển PD thành score

```python
import numpy as np


def probability_to_score(
    probability_of_default: np.ndarray,
    *,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pdo: float = 20.0,
    min_score: float | None = 300.0,
    max_score: float | None = 850.0,
) -> np.ndarray:
    pd_array = np.asarray(
        probability_of_default,
        dtype=float,
    )

    epsilon = 1e-9
    pd_array = np.clip(
        pd_array,
        epsilon,
        1.0 - epsilon,
    )

    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)

    odds = (1.0 - pd_array) / pd_array
    score = offset + factor * np.log(odds)

    if min_score is not None:
        score = np.maximum(score, min_score)

    if max_score is not None:
        score = np.minimum(score, max_score)

    return score
```

## 23.7 Phân rã điểm

```python
def score_contributions(
    woe_frame: pd.DataFrame,
    *,
    coefficients: np.ndarray,
    intercept: float,
    feature_names: list[str],
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pdo: float = 20.0,
) -> pd.DataFrame:
    if len(coefficients) != len(feature_names):
        raise ValueError(
            "Số coefficient phải bằng số feature."
        )

    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)

    contributions = pd.DataFrame(
        index=woe_frame.index
    )

    for feature_name, coefficient in zip(
        feature_names,
        coefficients,
    ):
        contributions[feature_name] = (
            -factor
            * coefficient
            * woe_frame[feature_name]
        )

    contributions["base_points"] = (
        offset - factor * intercept
    )

    contributions["raw_score"] = (
        contributions["base_points"]
        + contributions[feature_names].sum(axis=1)
    )

    return contributions
```

## 23.8 Kiểm tra tính nhất quán

Score tính từ PD phải gần bằng score tính từ contribution, trước khi rounding hoặc clipping.

```python
score_from_pd = probability_to_score(
    model.predict_proba(X_train)[:, 1],
    min_score=None,
    max_score=None,
)

contribution_table = score_contributions(
    X_train,
    coefficients=model.coef_[0],
    intercept=float(model.intercept_[0]),
    feature_names=feature_names,
)

np.testing.assert_allclose(
    score_from_pd,
    contribution_table["raw_score"].to_numpy(),
    rtol=1e-8,
    atol=1e-8,
)
```

---

# 24. Những lỗi thường gặp

## 24.1 Đảo dấu WoE nhưng không đảo cách diễn giải

Hai công thức:

$$
\ln\left(\frac{Good}{Bad}\right)
$$

và:

$$
\ln\left(\frac{Bad}{Good}\right)
$$

khác dấu. Cần ghi rõ convention trong code và tài liệu.

## 24.2 Fit WoE trên toàn bộ dataset

Đây là leakage vì WoE sử dụng target.

## 24.3 Quá nhiều bin

Hậu quả:

- bin nhỏ;
- zero Good/Bad;
- WoE cực đoan;
- IV ảo;
- score không ổn định.

## 24.4 Tối đa hóa IV mà bỏ qua stability

Một binning có IV cao trên train có thể giảm mạnh trên validation hoặc out-of-time sample.

## 24.5 Không tách missing

Missing có thể là tín hiệu quan trọng.

## 24.6 Dùng IV như tiêu chí duy nhất

Cần xem thêm:

- nghiệp vụ;
- leakage;
- multicollinearity;
- stability;
- fairness;
- missing rate;
- khả năng thu thập khi inference.

## 24.7 Giữ các biến tương quan quá cao

Ví dụ:

- số lần quá hạn 30 ngày;
- số lần quá hạn 60 ngày;
- số lần quá hạn 90 ngày;
- tổng số lần quá hạn.

Các biến có thể chứa thông tin trùng lặp, làm hệ số không ổn định.

Có thể kiểm tra:

- correlation giữa WoE features;
- Variance Inflation Factor;
- coefficient sign;
- coefficient stability qua sample;
- backward/forward selection.

## 24.8 Ép monotonicity không có cơ sở

Một số quan hệ thật sự phi đơn điệu.

## 24.9 Diễn giải WoE là nhân quả

WoE chỉ mô tả association trong dữ liệu.

## 24.10 Dùng score range làm bằng chứng calibration

Score `700` không có ý nghĩa phổ quát nếu không biết:

- Base Score;
- Base Odds;
- PDO;
- target definition;
- thời gian;
- population;
- calibration.

## 24.11 Tự đặt cutoff rồi gọi là model output

Model tạo `PD` và score. Rule approve/reject là policy, trừ khi dự án có dữ liệu và hàm
mục tiêu riêng để tối ưu quyết định.

---

# 25. Checklist triển khai cho dự án CreditScoring

## 25.1 Định nghĩa dữ liệu

- [ ] Xác định thời điểm ra quyết định.
- [ ] Xác định observation window.
- [ ] Xác định performance window.
- [ ] Định nghĩa Good và Bad.
- [ ] Loại biến chỉ xuất hiện sau quyết định.
- [ ] Kiểm tra duplicate và data quality.
- [ ] Xác định missing và special values.

## 25.2 Chia dữ liệu

- [ ] Tách train, validation và test trước khi binning.
- [ ] Stratify nếu random split.
- [ ] Dùng out-of-time split nếu có timestamp.
- [ ] Chỉ dùng test một lần cho báo cáo cuối.

## 25.3 Binning

- [ ] Tạo pre-bin.
- [ ] Đặt minimum bin size.
- [ ] Đặt minimum Good/Bad count.
- [ ] Merge bin có zero count.
- [ ] Kiểm tra monotonicity.
- [ ] Kiểm tra ý nghĩa nghiệp vụ.
- [ ] Giữ missing/special riêng khi cần.
- [ ] Lưu cut point.

## 25.4 WoE và IV

- [ ] Ghi rõ convention Good/Bad.
- [ ] Ghi rõ smoothing.
- [ ] Tính WoE chỉ trên train.
- [ ] Transform validation/test bằng mapping train.
- [ ] Tính IV.
- [ ] Điều tra biến có IV quá cao.
- [ ] Loại biến IV thấp nếu không có giá trị khác.
- [ ] Kiểm tra stability của WoE.

## 25.5 Logistic Regression

- [ ] Kiểm tra coefficient sign.
- [ ] Kiểm tra multicollinearity.
- [ ] Regularization.
- [ ] Đánh giá AUC/Gini/KS.
- [ ] Đánh giá calibration.
- [ ] Đánh giá theo segment.
- [ ] Kiểm tra fairness và proxy risk.

## 25.6 Scorecard

- [ ] Chọn Base Score.
- [ ] Chọn Base Odds.
- [ ] Chọn PDO.
- [ ] Tính Factor và Offset.
- [ ] Kiểm tra score từ PD bằng score từ contribution.
- [ ] Xác định rounding.
- [ ] Xác định clipping.
- [ ] Tạo bảng point theo bin.
- [ ] Tạo adverse reason.

## 25.7 Business rule

- [ ] Phân tích approval rate theo cutoff.
- [ ] Phân tích Bad Rate trong nhóm approve.
- [ ] Ghi rõ cutoff là policy, không phải model output.
- [ ] Nếu tối ưu cutoff, ghi rõ objective và assumption.
- [ ] Có vùng manual review nếu cần.

## 25.8 Monitoring

- [ ] Feature PSI.
- [ ] Score PSI.
- [ ] Missing rate.
- [ ] Unseen category rate.
- [ ] Approval rate.
- [ ] Bad Rate sau label maturity.
- [ ] AUC/Gini/KS theo thời gian.
- [ ] Calibration theo thời gian.
- [ ] WoE drift.
- [ ] Quy tắc cảnh báo và retraining.

---

# 26. Kết luận

Weight of Evidence là một phép biến đổi có giám sát, sử dụng thông tin Good/Bad trong
training set để biểu diễn mức độ bằng chứng rủi ro của từng bin.

Cốt lõi của WoE:

$$
WoE_i=
\ln\left(
\frac{P(Bin_i\mid Good)}
{P(Bin_i\mid Bad)}
\right)
$$

Quy trình chính:

```text
Raw value
      ↓
Bin
      ↓
Good/Bad distribution
      ↓
WoE
      ↓
Logistic Regression
      ↓
PD
      ↓
Credit Score
      ↓
Feature point contributions
```

Điểm mạnh quan trọng nhất của WoE Scorecard không phải chỉ là tạo ra một con số dự báo.
Nó tạo ra một hệ thống chấm điểm có cấu trúc:

- mỗi giá trị được map vào một bin rõ ràng;
- mỗi bin có bằng chứng thống kê;
- mỗi đặc trưng có đóng góp điểm cụ thể;
- tổng điểm có thể kiểm tra chính xác;
- quyết định có thể được giải thích;
- mô hình có thể được giám sát theo thời gian.

Trong phạm vi dự án CreditScoring, một đầu ra hoàn chỉnh nên gồm:

1. Bảng binning và WoE cho từng đặc trưng.
2. IV và lý do chọn hoặc loại biến.
3. Logistic Regression đã được đánh giá.
4. PD cho từng hồ sơ.
5. Credit Score theo Base Score, Base Odds và PDO.
6. Bảng đóng góp điểm theo từng đặc trưng.
7. Adverse reasons.
8. Rule approve, reject hoặc manual review được ghi rõ là business policy.
9. AUC, Gini, KS và calibration.
10. PSI và kế hoạch monitoring.

Như vậy, dự án không còn là việc truyền tham số vào một mô hình rồi nhận một đầu ra ML.
Nó trở thành một pipeline Credit Scoring có khả năng giải thích, kiểm toán, triển khai và
theo dõi sau triển khai.

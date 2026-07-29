# Báo cáo chi tiết về Weight of Evidence (WoE)

# 1. Giới thiệu

**Weight of Evidence (WoE)** là một kỹ thuật biến đổi đặc trưng (feature
transformation) được sử dụng rất rộng rãi trong các mô hình **Credit
Scoring** truyền thống.

Thay vì đưa trực tiếp giá trị gốc của đặc trưng vào Logistic Regression,
WoE chuyển đổi mỗi giá trị thành một đại lượng biểu diễn **mức độ bằng
chứng thống kê** cho thấy khách hàng thuộc nhóm **Good (không vỡ nợ)**
hay **Bad (vỡ nợ)**.

Pipeline tổng quát:

``` text
Dữ liệu gốc
      ↓
Binning
      ↓
Weight of Evidence (WoE)
      ↓
Logistic Regression
      ↓
Probability of Default (PD)
      ↓
Credit Score
```

------------------------------------------------------------------------

# 2. Tại sao không dùng trực tiếp giá trị gốc?

Ví dụ đặc trưng **Debt Ratio**.

  Khách hàng     Debt Ratio Default
  ------------ ------------ ---------
  A                     10% No
  B                     15% No
  C                     25% No
  D                     35% Yes
  E                     45% Yes
  F                     70% Yes
  G                     90% Yes

Quan hệ giữa Debt Ratio và xác suất vỡ nợ thường **không tuyến tính**.

Ví dụ:

-   10% → 20%: gần như không tăng rủi ro.

-   20% → 40%: tăng vừa phải.

-   40% → 60%: tăng nhanh.

-   60%: cực kỳ rủi ro.

WoE chuyển các vùng phi tuyến này thành các giá trị có ý nghĩa thống kê.

------------------------------------------------------------------------

# 3. Bước 1: Chia thành các bin

  Debt Ratio   Bin
  ------------ -----
  0--20%       A
  20--40%      B
  40--60%      C
  \>60%        D

------------------------------------------------------------------------

# 4. Bước 2: Đếm số Good và Bad

Giả sử toàn bộ tập huấn luyện có

-   Good = 900
-   Bad = 500

Trong từng bin:

  Bin     Good   Bad
  ----- ------ -----
  A        400    20
  B        300    80
  C        150   170
  D         50   230

------------------------------------------------------------------------

# 5. Bước 3: Chuyển sang tỷ lệ

  Bin               %Good             %Bad
  ----- ----------------- ----------------
  A       400/900 = 0.444   20/500 = 0.040
  B                 0.333            0.160
  C                 0.167            0.340
  D                 0.056            0.460

------------------------------------------------------------------------

# 6. Bước 4: Tính WoE

Công thức:

$$
WoE_i=\ln\left(\frac{\%Good_i}{\%Bad_i}\right)
$$

## Ví dụ Bin A

$$
WoE=\ln\left(\frac{0.444}{0.040}\right)
=\ln(11.1)
\approx2.41
$$

Ý nghĩa:

Bin này chứa tỷ lệ khách hàng **Good** cao hơn rất nhiều so với **Bad**.

## Ví dụ Bin D

$$
WoE=\ln\left(\frac{0.056}{0.460}\right)
=\ln(0.122)
\approx-2.10
$$

Ý nghĩa:

Bin này chứa nhiều khách hàng **Bad** hơn **Good**.

Kết quả:

  Debt Ratio       WoE
  ------------ -------
  0--20%         +2.41
  20--40%        +0.73
  40--60%        -0.71
  \>60%          -2.10

------------------------------------------------------------------------

# 7. Thay thế dữ liệu gốc

Thay vì

``` text
Debt Ratio = 73%
```

mô hình sẽ nhận

``` text
Debt Ratio WoE = -2.10
```

------------------------------------------------------------------------

# 8. Logistic Regression sử dụng WoE như thế nào?

Logistic Regression tính

$$
z=\beta_0+\sum_i \beta_i WoE_i
$$

Sau đó tính xác suất vỡ nợ

$$
PD=\frac{1}{1+e^{-z}}
$$

Ví dụ:

-   Debt Ratio WoE = -2.10
-   $$\beta=-0.8$$

Đóng góp của đặc trưng

$$
-0.8\times(-2.10)=1.68
$$

Giá trị này làm tăng log-odds của việc vỡ nợ.

------------------------------------------------------------------------

# 9. Khả năng giải thích (Explainability)

Khách hàng A

  Đặc trưng         WoE   Điểm đóng góp
  -------------- ------ ---------------
  Income           +1.5             +40
  Debt Ratio       +2.4             +30
  Late Payment     +1.8             +50

Điểm cuối

    600 + 40 + 30 + 50 = 720

Khách hàng B

  Đặc trưng         WoE   Điểm đóng góp
  -------------- ------ ---------------
  Income           -0.6             -15
  Debt Ratio       -2.1             -45
  Late Payment     -2.5             -80

Điểm cuối

    600 -15 -45 -80 = 460

Ngân hàng có thể giải thích:

-   Debt Ratio cao làm giảm 45 điểm.
-   Nhiều lần trả chậm làm giảm 80 điểm.

------------------------------------------------------------------------

# 10. WoE cho dữ liệu phân loại

  Nghề nghiệp     Good   Bad    WoE
  ------------- ------ ----- ------
  Government       200    10   +2.2
  Teacher          180    20   +1.5
  Student           60    80   -0.7
  Unemployed        30   120   -2.0

Thay vì One-Hot Encoding, mỗi nhóm được thay bằng mức độ rủi ro lịch sử.

------------------------------------------------------------------------

# 11. Information Value (IV)

IV thường đi cùng WoE để đánh giá sức mạnh của đặc trưng.

$$
IV=\sum_i(\%Good_i-\%Bad_i)\times WoE_i
$$

  IV           Ý nghĩa
  ------------ ----------------------------------
  \<0.02       Không có giá trị dự báo
  0.02--0.10   Yếu
  0.10--0.30   Trung bình
  0.30--0.50   Mạnh
  \>0.50       Quá mạnh, có thể bị data leakage

------------------------------------------------------------------------

# 12. Ưu điểm

-   Biểu diễn được quan hệ phi tuyến.
-   Xử lý tốt dữ liệu phân loại.
-   Dễ giải thích.
-   Ổn định theo thời gian.
-   Rất phù hợp với Logistic Regression.
-   Được chấp nhận rộng rãi trong ngành tài chính.

------------------------------------------------------------------------

# 13. Nhược điểm

-   Cần bước binning trước.
-   Chất lượng phụ thuộc vào cách chia bin.
-   Chủ yếu dành cho Scorecard truyền thống.
-   Ít lợi ích với các mô hình cây như LightGBM hoặc XGBoost.

------------------------------------------------------------------------

# 14. Tổng kết

WoE không chỉ là một phép chuẩn hóa dữ liệu mà là một phép biến đổi mang
ý nghĩa thống kê. Mỗi giá trị WoE biểu diễn mức độ bằng chứng rằng một
khách hàng thuộc nhóm Good hay Bad dựa trên dữ liệu lịch sử. Kết hợp với
Logistic Regression, WoE tạo nên các mô hình Credit Scorecard có khả
năng giải thích cao, dễ kiểm toán và đáp ứng yêu cầu của ngành tài
chính.

# Report #6 — Give Me Some Credit

**Link:** https://www.kaggle.com/competitions/GiveMeSomeCredit
**Host:** Credit Fusion · Featured Prediction Competition · 2011
**Giải thưởng:** $5,000 ($3,000 / $1,500 / $500)
**Dữ liệu:** 14.47 MB · 4 file · 150,000 dòng train
**Metric:** AUC

> Ghi chú nguồn: nội dung Overview lấy nguyên văn từ trang. File `Data Dictionary.xls` không đọc được qua fetch; định nghĩa biến đối chiếu từ nguồn thứ cấp và ghi rõ ở dưới.

## Vì sao đây là bộ dữ liệu để tập tay tuần này

Một file CSV 7 MB, 10 feature, không cần join, chạy trên laptop trong vài giây. Chạy hết vòng EDA → WoE → scorecard → PSI trong một ngày. Home Credit Default Risk 2.68 GB / 10 bảng thì không.

Nó cũng là bộ dữ liệu credit scoring "sạch" nhất về mặt khái niệm: một target rõ ràng, mười biến đều là biến chuẩn ngành, không có leakage.

## Mô tả bài toán (nguyên văn)

> "Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit.
>
> Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.
>
> The goal of this competition is to build a model that borrowers can use to help make the best financial decisions."

Câu cuối đáng chú ý: mô hình để **người vay** dùng ra quyết định tài chính, không chỉ để ngân hàng sàng lọc.

## Dữ liệu

File: `cs-training.csv`, `cs-test.csv`, `sampleEntry.csv`, `Data Dictionary.xls`.

### Biến

| Biến | Mô tả | Kiểu |
|---|---|---|
| **SeriousDlqin2yrs** | **TARGET** — khách đã trải qua quá hạn 90 ngày trở lên | Y/N |
| RevolvingUtilizationOfUnsecuredLines | Tổng dư nợ thẻ tín dụng + hạn mức cá nhân (trừ BĐS, trừ nợ trả góp như vay mua xe) chia cho tổng hạn mức | % |
| age | Tuổi người vay | integer |
| NumberOfTime30-59DaysPastDueNotWorse | Số lần quá hạn 30–59 ngày (không nặng hơn) trong 2 năm qua | integer |
| DebtRatio | Nghĩa vụ trả nợ hàng tháng + cấp dưỡng + chi phí sinh hoạt, chia cho thu nhập gộp hàng tháng | % |
| MonthlyIncome | Thu nhập hàng tháng | real |
| NumberOfOpenCreditLinesAndLoans | Số khoản vay và hạn mức tín dụng đang mở | integer |
| NumberOfTimes90DaysLate | Số lần quá hạn 90 ngày trở lên | integer |
| NumberRealEstateLoansOrLines | Số khoản vay thế chấp BĐS, gồm cả hạn mức home equity | integer |
| NumberOfTime60-89DaysPastDueNotWorse | Số lần quá hạn 60–89 ngày (không nặng hơn) trong 2 năm qua | integer |
| NumberOfDependents | Số người phụ thuộc (không tính bản thân) | integer |

> `SeriousDlqin2yrs`, `RevolvingUtilizationOfUnsecuredLines`, `DebtRatio` xác nhận nguyên văn từ data dictionary qua nguồn thứ cấp; các biến còn lại theo mô tả chuẩn của bộ dữ liệu.

Bad rate: ~**6.7%**.

Định nghĩa target trùng chuẩn Basel (90+ DPD) — xem [02-label-target-den-tu-dau.md](../00-tong-quan/02-label-target-den-tu-dau.md).

## Bốn nhóm feature — đúng khung lý thuyết

Mười biến này chính là bốn trụ cột kinh điển của credit scoring, dùng làm checklist khi khảo sát dữ liệu dự án:

| Nhóm | Biến | Ý nghĩa |
|---|---|---|
| **Delinquency history** | 3 biến đếm quá hạn 30-59 / 60-89 / 90+ | Hành vi quá khứ — dự báo mạnh nhất |
| **Utilization / leverage** | `RevolvingUtilization`, `DebtRatio` | Mức căng thẳng tài chính hiện tại |
| **Capacity** | `MonthlyIncome`, `NumberOfDependents` | Khả năng trả |
| **Demographic / exposure** | `age`, số khoản vay đang mở, số khoản BĐS | Bối cảnh |

Ba biến delinquency thường là ba biến IV cao nhất. Bài học chung: **hành vi trả nợ quá khứ dự báo hành vi trả nợ tương lai tốt hơn mọi thông tin nhân khẩu học.**

## Các bẫy dữ liệu đã biết (rất tốt để tập xử lý)

1. **`age` = 0** — một dòng. Không thể có. Loại hoặc gán median.
2. **`RevolvingUtilizationOfUnsecuredLines` > 1** — hàng nghìn dòng, có giá trị tới ~50,708. Trên 100% là có thật (vượt hạn mức), nhưng > 10 thì là lỗi. Cap ở percentile 99, hoặc tách bin riêng cho nhóm > 1.
3. **`DebtRatio`** giá trị cực lớn (tới ~329,664), liên quan tới các dòng `MonthlyIncome` bị NaN. Xem `DebtRatio` cùng cờ `MonthlyIncome_isna`.
4. **`MonthlyIncome`** missing ~19.8%. Bài tập tốt: bin missing riêng, xem WoE — thu nhập không khai báo có phải tín hiệu rủi ro?
5. **`NumberOfDependents`** missing ~2.6%.
6. **`NumberOfTime30-59DaysPastDue` = 96 và 98** — giá trị mã hóa đặc biệt, giống hệt `DAYS_EMPLOYED = 365243` của Home Credit. Xuất hiện ở cả ba biến delinquency, trên cùng một tập dòng. Xử lý: tạo cờ rồi thay NaN.

Bẫy số 6 là bài tập lý tưởng cho quy trình anomaly ở [04-eda-playbook.md](../00-tong-quan/04-eda-playbook.md).

## Mức hiệu năng tham chiếu

| Cách làm | AUC (private LB) |
|---|---|
| Logistic Regression thô | ~0.80 |
| Logistic Regression + binning/WoE + xử lý outlier | ~0.86 |
| GBDT có tuning | ~0.866 |
| Top leaderboard | ~0.8696 |

Khoảng cách giữa "làm cẩn thận bằng LR" và "top leaderboard" chỉ ~0.01 AUC. **Bài học: trên bộ ít feature, xử lý dữ liệu cẩn thận có giá trị hơn mô hình phức tạp.**

AUC ở đây (~0.86) cao hơn Home Credit Default Risk (~0.80) vì ba biến delinquency là bureau data trực tiếp và rất mạnh, không phải vì bài toán dễ hơn.

## Bài tập đề xuất cho tuần này

```
Ngày 2: EDA
  - bad rate, missing, describe()
  - phát hiện age=0, utilization>10, delinquency=96/98
  - bad rate theo decile của cả 10 biến
Ngày 3: baseline
  - LogisticRegression (impute median + scale) -> AUC
  - LightGBM                                   -> AUC
Ngày 4: WoE
  - tree-based binning cho 10 biến
  - bảng IV, kiểm tra WoE đơn điệu
  - lặp lại thí nghiệm equal-width vs tree binning của Report #4
Ngày 5: scorecard
  - LR trên WoE -> scorecard 300-850
  - cutoff theo approval rate 60% / 70% / 80% -> bad rate tương ứng
  - PSI giữa nửa đầu và nửa sau tập train
```

Tiêu chí xong: in được bảng `(feature, bin, WoE, IV, điểm)`, và có hàm `psi(expected, actual, bins)` tái dùng được.

## Hạn chế cần biết

- **Không có cột thời gian** → không tập được out-of-time split, không tính được gini theo kỳ.
- Chỉ 10 feature, không có bảng phụ → không tập được aggregation depth 1/2.
- Dữ liệu 2011, phân phối không giống thị trường Việt Nam hiện tại.
- Đây là dữ liệu **bureau-heavy**; thị trường thiếu bureau data sẽ không có ba biến delinquency mạnh như vậy.

Vì vậy: dùng để tập **quy trình** → Home Credit Default Risk tập **kỹ thuật dữ liệu** → Model Stability tập **validation theo thời gian**.

## Liên quan
- Định nghĩa label: [02-label-target-den-tu-dau.md](../00-tong-quan/02-label-target-den-tu-dau.md)
- Bộ dữ liệu bước tiếp: [Report #7](07-comp-home-credit-default-risk.md)
- Kế hoạch tuần: [07-ke-hoach-tuan-checklist.md](../00-tong-quan/07-ke-hoach-tuan-checklist.md)

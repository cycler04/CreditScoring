# Report #3 — Credit Risk EDA: Defaults, Segments & Trends (Part 1)

**Link:** https://www.kaggle.com/code/beatafaron/credit-risk-eda-defaults-segments-trends-1
**Tác giả:** Beata Faron
**Source GitHub:** https://github.com/BeataFaron/credit-risk-psi-scorecards (`notebooks/credit-risk-eda-defaults-segments-trends-1.ipynb`)
**Dữ liệu:** LendingClub 2014–2018, ~2.03 triệu khoản vay
**Vai trò:** Part 1 của series 3 notebook

> Ghi chú nguồn: tải được file `.ipynb` gốc từ GitHub. **Mọi code và con số dưới đây là nguyên văn.**

## Series 3 notebook

| Phần | Nội dung | Report |
|---|---|---|
| 1 | EDA, feature selection, so sánh 4 mô hình | file này |
| 2 | WoE + Logistic Regression + Scorecard | [Report #4](04-nb-credit-risk-eda-woe-scorecard-2.md) |
| 3 | PSI — giám sát drift | không có trong task.txt, nhưng nên đọc |

Đây là series duy nhất trong task.txt đi hết vòng đời: EDA → scorecard → monitoring. Đúng khung của một dự án credit scoring thật.

## Khung nghiệp vụ (nguyên văn, đáng đọc)

Notebook mở đầu bằng bối cảnh chứ không bằng code:

> "You've joined the credit risk team at LendingClub, a leading US peer-to-peer lending platform."

Bốn mục tiêu đặt ra:
1. EDA các mẫu rủi ro ở mức khách hàng.
2. Xây mô hình PD bằng WoE + Logistic Regression.
3. Dựng behavioral scorecard giải thích được.
4. Giám sát ổn định dài hạn bằng PSI.

Ba khái niệm được định nghĩa ngay: PD, WoE, PSI. Và phần Basel II: yêu cầu đủ vốn, supervisory review, minh bạch thị trường; retail exposure cần dự trữ vốn 75% tổng exposure theo Standardised Approach, có F-IRB / A-IRB nâng cao.

**Đây là mẫu tốt để copy:** mở đầu tài liệu mô hình bằng bối cảnh nghiệp vụ và khung pháp lý, không bằng `import pandas`.

## Quy trình thực hiện

### 1. Định nghĩa target (nguyên văn)

```python
loan_status_mapping = {
    'Fully Paid': 1, 'Current': 1, 'In Grace Period': 1,
    'Late (16-30 days)': 0, 'Late (31-120 days)': 0,
    'Charged Off': 0, 'Default': 0
}
df['loan_status_binary'] = df['loan_status'].map(loan_status_mapping)
```

Phân phối: **1,704,881 non-default (1)** vs **325,071 default (0)** — lớp thiểu số ~16%.

Ba điểm cần chú ý (đã phân tích ở [02-label-target-den-tu-dau.md](../00-tong-quan/02-label-target-den-tu-dau.md#25-hai-cách-map-label-trong-thực-tế--ví-dụ-đối-chiếu)):
- Quy ước **ngược** thông lệ: 1 = good, 0 = bad.
- Ngưỡng bad là **16+ DPD**, không phải 90+ DPD theo Basel.
- `Current` gán good dù khoản vay chưa chạy hết đời → label dạng snapshot, không theo performance window.

### 2. Missing values — ngưỡng 51%

```python
missing = missing_data_summary(df, 51)      # bảng % missing
df = df.drop(columns=missing.iloc[:,0].tolist())
```
Bỏ thẳng mọi cột missing > 51%, có biểu đồ top 10 cột missing nhiều nhất. Lý do notebook đưa ra: giảm nhiễu; cột còn thiếu vừa phải thì impute.

⚠️ Thiếu bước kiểm tra WoE của nhóm missing trước khi bỏ. Trong tín dụng, "thiếu dữ liệu" thường tự nó mang tín hiệu.

### 3. Làm sạch biến object (nguyên văn, mẫu tốt)

```python
df = df.apply(lambda col: col.str.strip() if col.dtypes == 'object' else col)

df['issue_d']            = pd.to_datetime(df['issue_d'], format='%Y-%m-%d')
df['earliest_cr_line']   = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')

df['int_rate%']   = pd.to_numeric(df['int_rate'].str.strip('%'))
df['revol_util%'] = pd.to_numeric(df['revol_util'].str.strip('%'))

df.debt_settlement_flag = np.where(df.debt_settlement_flag == 'Y', 1, 0)
df.term_36_months       = np.where(df.term == '36 months', 1, 0)
df['emp_length'] = pd.to_numeric(
    df['emp_length'].fillna('').str.replace('<','',regex=False).str[:2].str.strip(),
    errors='coerce')

df.drop({'title','zip_code','pymnt_plan','emp_title','int_rate','revol_util','url'},
        axis=1, inplace=True)
```
Điển hình cho dữ liệu tín dụng thật: phần trăm lưu dạng chuỗi `"13.5%"`, kỳ hạn lưu `"36 months"`, thâm niên lưu `"< 1 year"` / `"10+ years"`. Luôn phải parse thủ công.

### 4. Biến phân loại — nhận xét (nguyên văn)

> `sub_grade` đã nằm trong `grade` → thừa.
> `home_ownership` lệch nặng: `MORTGAGE` và `RENT` áp đảo, `OWN` và `OTHER` rất ít → cân nhắc gộp.
> `purpose` có nhiều nhóm hiếm (`medical`, `vacation`, `wedding`, `renewable_energy`, `educational`) → gộp vào `other`.
> `addr_state` 51 giá trị, quá chi tiết → gộp theo vùng, cụm theo mức rủi ro, hoặc bỏ.

### 5. WoE & IV cho biến phân loại

Công thức dùng trong notebook (theo quy ước 1 = good = "event", 0 = bad = "non-event"):

```
WoE = ln( %non-event / %event )
IV  = Σ (%non-event − %event) × WoE
```

Bảng ngưỡng IV (nguyên văn):

| IV | Ý nghĩa |
|---|---|
| < 0.02 | Not Predictive |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| 0.3+ | Strong |
| > 0.5 | Suspiciously strong — có thể overfit hoặc lỗi dữ liệu |

NaN được xử lý thành **một category riêng** (`df[col].fillna('NaN')`) — đúng cách làm chuẩn ngành.

Ứng dụng ngay:
```python
df.drop({'application_type','initial_list_status','addr_state','purpose'},
        axis=1, inplace=True)          # IV < 0.02

home_ownership_mapping = {'ANY':'other','MORTGAGE':'other','NONE':'other',
                          'RENT':'rent','OWN':'own'}
```
Gộp `ANY`/`MORTGAGE`/`NONE` vì **WoE của chúng gần nhau** — không gộp theo cảm tính. Đây là kỹ thuật đáng copy nhất của notebook.

### 6. Feature từ ngày tháng (nguyên văn)

```python
today = pd.to_datetime("today")
df['last_pymnt_d'].fillna(today, inplace=True)

df['loan_age']                    = (today - df['issue_d']).dt.days
df['credit_history_length']       = (df['issue_d'] - df['earliest_cr_line']).dt.days
df['time_since_last_payment']     = (today - df['last_pymnt_d']).dt.days
df['time_since_last_credit_pull'] = (today - df['last_credit_pull_d']).dt.days
df['issue_year']  = df['issue_d'].dt.year
df['issue_month'] = df['issue_d'].dt.month
df['recent_payment']     = (df['time_since_last_payment'] <= 30).astype(int)
df['recent_credit_pull'] = (df['time_since_last_credit_pull'] <= 90).astype(int)
```

Ý tưởng hiệu số ngày đúng, nhưng **mốc `today` sai** — phải là ngày quyết định của từng hồ sơ, không phải ngày chạy script. Dùng `today` khiến feature đổi giá trị mỗi lần chạy lại và rò rỉ thông tin tương lai.

### 7. Feature selection ba tầng

**Tầng 1 — bỏ điểm số ngoài (chủ ý, đáng học):**
```python
external_scores = {'last_fico_range_high','last_fico_range_low',
                   'fico_range_low','fico_range_high'}
df_final.drop(external_scores, axis=1, inplace=True)
```
Bỏ FICO để mô hình học từ dữ liệu thô thay vì học lại điểm của bên khác. Đây là quyết định nghiệp vụ có ý thức.

**Tầng 2 — tương quan:**
```python
threshold = 0.8
# trong mỗi cặp |corr| > 0.8, bỏ biến có tương quan với target thấp hơn
```
Và tạo ratio thay vì bỏ hẳn:
```python
df_final['loan_amnt_div_instlmnt'] = df_final['loan_amnt'] / df_final['installment']
```

**Tầng 3 — RandomForest + SelectFromModel:**
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
model = SelectFromModel(rf, prefit=True, threshold="mean")
```

Kết quả 12 feature được chọn (nguyên văn):
```
out_prncp, total_rec_prncp, total_rec_int, total_rec_late_fee,
recoveries, last_pymnt_amnt, debt_settlement_flag, int_rate%,
loan_age, time_since_last_payment, time_since_last_credit_pull,
loan_amnt_div_instlmnt
```

### 8. So sánh 4 mô hình

```python
lr_model = LogisticRegression(random_state=42)
rf_soft  = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
nn_model = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=300, solver='adam')
```
Đánh giá bằng confusion matrix, accuracy, recall, **F2**, AUC. Chọn F2 (ưu tiên recall gấp đôi precision) là hợp lý với bài toán rủi ro tín dụng: bỏ sót khách xấu đắt hơn từ chối nhầm khách tốt.

Kết luận notebook: Neural Network và Random Forest cho kết quả tốt nhất; đề xuất thử nhiều threshold để cân bằng FP/FN.

---

## ⚠️ CẢNH BÁO QUAN TRỌNG — Leakage

**10 trên 12 feature được chọn ở tầng 3 là leakage.** Chúng chỉ tồn tại hoặc chỉ có giá trị **sau khi khoản vay đã giải ngân và chạy được một thời gian** — tức sau thời điểm quyết định cho vay.

| Feature được chọn | Vì sao leak |
|---|---|
| `recoveries` | Chỉ ≠ 0 khi khoản vay **đã** charged-off — gần như chính là label |
| `total_rec_prncp`, `total_rec_int` | Tổng gốc/lãi đã thu, tích lũy suốt đời khoản vay |
| `total_rec_late_fee` | Phí phạt trễ hạn đã thu — trực tiếp là hành vi quá hạn |
| `out_prncp` | Dư nợ còn lại hiện tại |
| `last_pymnt_amnt` | Số tiền lần trả gần nhất |
| `time_since_last_payment` | Bao lâu rồi chưa trả — chính là DPD trá hình |
| `time_since_last_credit_pull` | Hoạt động sau giải ngân |
| `debt_settlement_flag` | Cờ đã thỏa thuận xử lý nợ xấu |
| `loan_age` | Vay càng lâu càng có cơ hội xấu |

Chỉ `int_rate%` và `loan_amnt_div_instlmnt` là dùng được cho application scorecard.

**Hệ quả:** kết quả mô hình của notebook (và AUC ~0.99 ở Part 2) **không phải năng lực dự báo**, mà là mô hình đọc lại kết cục từ dữ liệu hậu kiểm.

**Điều này không làm notebook mất giá trị**, nhưng phải đọc đúng cách:
- **Đáng học:** quy trình, code làm sạch, cách dùng WoE/IV để gộp category và lọc biến, khung nghiệp vụ Basel, cách trình bày.
- **Không đáng học:** danh sách feature cuối, con số AUC, và kết luận về mô hình nào tốt nhất.

**Cách sửa nếu dựng lại:** chốt thời điểm quyết định = `issue_d`, giữ **duy nhất** các cột có giá trị tại thời điểm đó (`loan_amnt`, `term`, `int_rate`, `grade`, `emp_length`, `annual_inc`, `dti`, `home_ownership`, `verification_status`, `earliest_cr_line`, `open_acc`, `revol_bal`, `revol_util`, `total_acc`, `delinq_2yrs`, `inq_last_6mths`, `pub_rec`), bỏ toàn bộ cột `total_rec_*`, `out_*`, `last_*`, `recoveries*`, `*_pymnt*`, `hardship_*`, `settlement_*`. AUC kỳ vọng khi đó khoảng **0.68–0.72**.

## Rút ra cho dự án

**Copy được ngay:**
1. Mở tài liệu mô hình bằng bối cảnh nghiệp vụ + khung pháp lý.
2. Hàm `missing_data_summary(df, threshold)`.
3. Mẫu parse chuỗi bẩn (`%`, `"36 months"`, `"< 1 year"`).
4. Dùng WoE/IV để (a) lọc biến IV < 0.02 và (b) quyết định gộp category theo WoE gần nhau.
5. Quy trình chọn biến ba tầng: bỏ điểm số ngoài → xử lý tương quan cao → RF importance.
6. Dùng F2 bên cạnh AUC.

**Sửa lại trước khi dùng:**
1. Ngưỡng default 16+ DPD → 90+ DPD.
2. Quy ước 1 = bad (không phải 1 = good).
3. Mốc thời gian `today` → `date_decision` từng hồ sơ.
4. **Rà leakage trước khi chạy feature selection**, không phải sau.
5. Split out-of-time thay vì `train_test_split(random_state=42)`.
6. Kiểm tra WoE của nhóm missing trước khi bỏ cột theo ngưỡng 51%.

## Liên quan
- Phần tiếp: [Report #4 — WoE & Scorecard](04-nb-credit-risk-eda-woe-scorecard-2.md)
- Chi tiết về leakage: [03-feature-den-tu-dau.md](../00-tong-quan/03-feature-den-tu-dau.md#37-feature-cấm-dùng-leakage)

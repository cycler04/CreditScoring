# Report #4 — Credit Risk EDA | WoE & Scorecard (Part 2)

**Link:** https://www.kaggle.com/code/beatafaron/credit-risk-eda-woe-scorecard-2
**Tác giả:** Beata Faron · huy chương đồng · 32 upvote
**Source GitHub:** https://github.com/BeataFaron/credit-risk-psi-scorecards (`notebooks/credit-risk-eda-woe-scorecard-2.ipynb`)
**Dữ liệu:** `df_2014-18_selected.csv` — output của [Part 1](03-nb-credit-risk-eda-defaults-segments-trends.md), 12 feature đã chọn
**Vai trò:** Part 2 của series 3

> Ghi chú nguồn: tải được `.ipynb` gốc từ GitHub. **Mọi code, công thức và con số dưới đây là nguyên văn.**

## Đây là notebook quan trọng nhất trong task.txt

Vì nó là notebook duy nhất đi hết quy trình **WoE → Logistic Regression → Scorecard → Cutoff → Risk grade**, tức là con đường chuẩn ngành để biến mô hình thành sản phẩm dùng được. Ba notebook còn lại dừng ở "AUC bao nhiêu".

## Phạm vi (nguyên văn)

> - Create a **behavioral scorecard**
> - Define **risk groups** (High, Medium, Low)
> - Assign **score ranges** (300–900) based on: statistical analysis (WOE, IV), business rules (cutoffs for default rates)

Và ba quy tắc gán điểm:
> 1. **WOE Transformation** — biến liên tục và phân loại được binning rồi chuyển sang WoE
> 2. **Logistic Regression** — tính PD rồi map sang điểm
> 3. **Score Scaling** — scale về 300–900 bằng
> $$\text{Score} = \text{Base Score} + \text{Factor} \times \log\left(\frac{1-PD}{PD}\right)$$
> với Base Score = điểm khởi đầu (ví dụ 300), Factor = độ dốc (ví dụ 20 điểm mỗi lần odds gấp đôi)

Đây chính là công thức scorecard chuẩn ngành. `Factor = 20 cho mỗi lần odds gấp đôi` = PDO 20.

## Nội dung theo mục

### Mục 2 — Binning: so sánh hai phương pháp

Notebook viết hai hàm và so sánh trực tiếp:

| Hàm | Cách | Ưu | Nhược |
|---|---|---|---|
| `bin_and_plot_woe_manual` | `pd.cut()` — equal width | Đơn giản, dễ hiểu | Có thể không bắt được pattern |
| `bin_and_plot_woe_tree` | `DecisionTreeClassifier(max_depth=4)` | Bám dữ liệu, bắt được pattern | Overfit nếu `max_depth` lớn |

Nguyên tắc về đơn điệu (nguyên văn):

> Weight of Evidence (WoE) should be monotonic with respect to the target variable... Monotonic WoE ensures more stable and interpretable relationships with the target, reduces the risk of overfitting, and improves the predictive power of scorecards.

### Kết quả so sánh — con số đáng nhớ nhất

Trên biến `total_rec_late_fee`: mean và std đều quanh 2, nhưng **max = 1,598.52** → có outlier nặng.

```
Total IV for total_rec_late_fee (equal-width binning): 0.0004
Total IV for total_rec_late_fee (decision-tree binning): 0.2404
```

Chênh **600 lần**. Nhận xét của notebook: equal-width dồn gần như toàn bộ dữ liệu vào một bin (khoảng −1.59 đến 319), trong khi decision tree tìm được ngưỡng chia có ý nghĩa ở khoảng 0.035.

**Bài học:** IV thấp **không** chứng minh biến vô dụng. Có thể chỉ là binning tồi. Luôn thử tree-based binning trước khi loại biến, và luôn kiểm tra outlier trước khi binning.

### Mục 3 — WoE encoding sang dummy

Chiến lược: mỗi bin thành một cột dummy, bỏ bin đầu làm reference (tránh dummy trap).

```python
for feature in features:
    woe_tree = bin_and_plot_woe_tree(df, feature, 'loan_status_binary', max_depth=4)
    df_prep, dropped_first = transform_to_dummy(df, feature, woe_tree, df_prep, dropped_first)

df_prep_dropped_first = df_prep.drop(columns=dropped_first["col_name"].tolist())
```
Bin bị bỏ được lưu lại trong `dropped_first` — cần thiết vì ở bước scorecard phải gán lại cho nó hệ số 0.

Kết quả: 12 feature → **135 cột dummy** (thấy ở `scores.values.reshape(135,1)`).

### Mục 4 — Logistic Regression

```python
X = df_prep_dropped_first.copy()
y = target
model = LogisticRegression()
model.fit(X, y)

summary_table = pd.DataFrame()
summary_table['feature_name']  = df_prep_dropped_first.columns
summary_table['coefficience']  = np.transpose(model.coef_)
summary_table.loc[-1] = ['intercept', model.intercept_[0]]
```

**Bẫy reproducibility được ghi lại ngay trong notebook (nguyên văn):**

```python
"""
Jupyter: scikit-learn 1.5.1
Kaggle:  scikit-learn 1.2.2 (Older version!)
This version mismatch is causing the huge difference in
logistic regression coefficients (intercept_ values).

model.intercept_ in kaggle  : array([-7.52160171])
model.intercept_ in jupyter : array([-3.48804904])
"""
!pip install --upgrade scikit-learn==1.5.1
```

Cùng code, cùng dữ liệu, khác version → intercept lệch hơn gấp đôi. Với scorecard, intercept quyết định toàn bộ mức điểm cơ sở. **Bài học: pin version thư viện trong `requirements.txt` và lưu model artifact, không lưu code rồi train lại.**

(Nguyên nhân kỹ thuật: sklearn đổi solver mặc định và cách xử lý hội tụ giữa các version; với dữ liệu tách gần hoàn hảo do leakage, hệ số chưa hội tụ ổn định.)

### Mục 5 — Scorecard

Gán hệ số 0 cho các bin reference đã bỏ, rồi scale toàn bộ về dải 300–900:

```python
new = pd.DataFrame({'feature_name': dropped_first["feature"] + "_" + dropped_first["dropped_bin"],
                    'coefficience': 0})
scorecard = pd.concat([summary_table, new]).reset_index()
scorecard['feature_original'] = scorecard['feature_name'].str.split('_(', regex=False).str[0]

min_sum = scorecard.groupby('feature_original')['coefficience'].min().sum()
max_sum = scorecard.groupby('feature_original')['coefficience'].max().sum()
max_score, min_score = 900, 300

scorecard['score_cal'] = scorecard['coefficience'] * (max_score-min_score)/(max_sum-min_sum)
scorecard['score_cal'][0] = (scorecard['coefficience'][0]-min_sum)/(max_sum-min_sum)*(max_score-min_score)+min_score
```

Logic: hồ sơ tệ nhất có thể (chọn bin xấu nhất ở mọi biến) = 300 điểm, tốt nhất có thể = 900 điểm. Intercept gánh phần offset. Sau đó kiểm tra lại bằng `min_check` / `max_check`.

Tính điểm cho từng hồ sơ = phép nhân ma trận:
```python
df_prep.insert(0, 'intercept', 1)
df_prep = df_prep[scorecard['feature_name'].values]
scores  = scorecard['score_cal'].values.reshape(135,1)
df_prep_scores = df_prep.dot(scores).astype(int)
```
Đây là cách triển khai đẹp: **scorecard cuối cùng chỉ là một vector 135 số**, deploy được bằng SQL hoặc Excel, không cần Python runtime.

### Mục 6 — ⚠️ Chuyển điểm sang xác suất (SAI)

```python
y_score = 1 / (1 + np.exp(-(df_prep_scores - min_score) / (max_score - min_score)))
```

Công thức này **không phải** nghịch đảo của phép scale ở mục 5. Nó chia cho `(900-300) = 600` như thể đó là hệ số scale của log-odds, trong khi 600 chỉ là độ rộng dải điểm. Kết quả `y_score` bị nén vào một dải hẹp quanh 0.5–0.73.

Nghịch đảo đúng phải là:
```python
PD = 1 / (1 + np.exp((score - base_score) / factor))     # factor = PDO / ln(2)
```

Vì sao vẫn ra AUC cao: AUC chỉ phụ thuộc **thứ tự**, và hàm sigmoid đơn điệu nên không đổi thứ tự. Nên AUC vẫn đúng, còn giá trị xác suất thì vô nghĩa. Nếu dùng PD cho pricing hay trích lập dự phòng thì sai hoàn toàn.

### Mục 7 — ⚠️ Cutoff bằng Youden's J

```python
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

optimal_score_cutoff = min_score + (max_score - min_score) * np.log(optimal_threshold/(1-optimal_threshold))
```

Hai vấn đề:
1. **Youden's J coi FP và FN có chi phí bằng nhau** — trong tín dụng thì không (xem [01-co-ban-cho-vay-va-credit-scoring.md](../00-tong-quan/01-co-ban-cho-vay-va-credit-scoring.md#11-bài-toán-kinh-doanh)). Cutoff đúng phải chọn theo lợi nhuận hoặc theo approval rate mục tiêu.
2. Công thức đổi ngược từ xác suất sang điểm dùng `min + range × ln(odds)` — không khớp với công thức thuận ở mục 6, và thiếu phép chia cho factor. Sẽ ra số điểm nằm ngoài dải 300–900.

Ý tưởng (chọn cutoff → chuyển sang ngưỡng điểm) thì đúng và cần thiết; công thức thì phải viết lại.

### Mục 8 — Risk category

```python
df_prep_scores['Risk_Category'] = np.where(df_prep_scores >= optimal_score_cutoff, "Good", "Bad")
```

Đây là bước biến mô hình thành quyết định. Trong thực tế nên chia nhiều hơn 2 nhóm (A/B/C/D/E hoặc Low/Medium/High như phần Scope đã nêu) để định giá theo rủi ro chứ không chỉ duyệt/từ chối.

## Tổng hợp cảnh báo

| Vấn đề | Mức độ | Ghi chú |
|---|---|---|
| Kế thừa 10/12 feature leak từ Part 1 | **Nghiêm trọng** | AUC báo cáo không phản ánh năng lực dự báo |
| Train trên toàn bộ dữ liệu, không có tập test | **Nghiêm trọng** | `model.fit(X, y)` rồi đánh giá trên chính `X` |
| Không split out-of-time | Nghiêm trọng | Dữ liệu 2014–2018 hoàn toàn có thể split theo năm |
| Công thức score → PD sai (mục 6) | Trung bình | AUC vẫn đúng; PD thì không dùng được |
| Công thức PD → score cutoff sai (mục 7) | Trung bình | |
| Youden's J làm cutoff | Trung bình | Sai về mặt kinh tế cho bài toán tín dụng |
| Version sklearn đổi kết quả | Đã được tác giả ghi nhận | Bài học tốt về reproducibility |

Notebook tự đánh giá `"0.9 - 1.0 → Excellent"` cho ROC-AUC. Với application scorecard, AUC > 0.9 gần như luôn có nghĩa là leakage, không phải xuất sắc.

## Rút ra cho dự án

**Copy được ngay (giá trị cốt lõi của notebook):**
1. Toàn bộ khung: binning → WoE → dummy (drop first) → LR → scale ra điểm → cutoff → risk grade.
2. So sánh equal-width vs tree-based binning, và bằng chứng IV 0.0004 → 0.2404.
3. Nguyên tắc WoE đơn điệu.
4. Lưu lại bin bị drop để gán hệ số 0 khi dựng scorecard.
5. Scorecard cuối = vector số → deploy được không cần Python.
6. Bài học pin version thư viện.

**Phải sửa khi dựng lại:**
1. Bỏ mọi feature leak trước, chỉ giữ biến có tại thời điểm quyết định.
2. Split out-of-time; không bao giờ đánh giá trên tập đã train.
3. Dùng công thức chuẩn: `Score = Base + Factor·ln(odds)`, `Factor = PDO/ln(2)`; và nghịch đảo `PD = 1/(1+exp((score−Base)/Factor))`.
4. Cutoff theo lợi nhuận hoặc approval rate mục tiêu, không dùng Youden's J.
5. Ràng buộc bin: mỗi bin ≥ 5% dữ liệu, WoE đơn điệu, 3–7 bin mỗi biến.
6. Kiểm tra dấu hệ số LR có hợp trực giác nghiệp vụ không.

## Liên quan
- Phần trước: [Report #3](03-nb-credit-risk-eda-defaults-segments-trends.md)
- Phần 3 (PSI, không có trong task.txt nhưng nên đọc): https://www.kaggle.com/code/beatafaron/monitor-feature-drift-with-psi-credit-risk-3
- Lý thuyết scorecard: [05-modeling-playbook.md](../00-tong-quan/05-modeling-playbook.md#52-pipeline-scorecard-woe--lr--đầy-đủ)

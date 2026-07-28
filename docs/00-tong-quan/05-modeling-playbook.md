# 5. Modeling playbook

## 5.1 Hai trường phái

| | WoE + Logistic Regression (scorecard) | GBDT (LightGBM / XGBoost / CatBoost) |
|---|---|---|
| AUC điển hình | thấp hơn 0.01–0.03 | cao nhất |
| Giải thích | Từng bin ra từng điểm, audit được | SHAP, gián tiếp |
| Missing / outlier | Nuốt trọn qua binning | Nuốt trọn (LightGBM xử lý NaN native) |
| Categorical | WoE encode | Native (LightGBM/CatBoost) |
| Monotonic theo nghiệp vụ | Ép được bằng binning | Cần `monotone_constraints` |
| Rủi ro overfit | Thấp | Cao, cần tuning |
| Thời gian dựng | Lâu (binning thủ công) | Nhanh |

Chọn: **làm cả hai**. LR là baseline giải thích được và là thứ đem đi thuyết phục nghiệp vụ; GBDT cho biết trần AUC của bộ dữ liệu. Nếu chênh lệch nhỏ → dùng scorecard. Nếu chênh lớn → có quan hệ phi tuyến/tương tác mà binning bỏ sót, quay lại sửa binning.

## 5.2 Pipeline scorecard (WoE + LR) — đầy đủ

### B1. Binning

Chia biến liên tục thành bin. Hai cách, notebook "Credit Risk EDA | WOE & Scorecard" so sánh trực tiếp:

| Cách | Ưu | Nhược |
|---|---|---|
| `pd.cut()` — equal width | Đơn giản, dễ hiểu | Nhạy outlier, bỏ sót pattern |
| `DecisionTreeClassifier` — tree-based | Bám dữ liệu, tối ưu theo target | Overfit nếu `max_depth` lớn |

Con số minh họa mạnh từ notebook đó, trên cùng một biến `total_rec_late_fee`:

```
IV với equal-width binning : 0.0004   → "không dự báo được"
IV với decision-tree binning: 0.2404   → "sức dự báo trung bình"
```

Cùng một biến, khác cách chia bin, kết luận ngược nhau hoàn toàn. Nguyên nhân: biến có outlier (max ≈ 1,598 trong khi mean ≈ 2), equal-width dồn 99.9% dữ liệu vào một bin.

Quy tắc binning thực hành:
- Mỗi bin ≥ 5% tổng số quan sát, và có ít nhất vài chục bad.
- 3–7 bin mỗi biến là đủ.
- Missing để thành **bin riêng**, không impute.
- WoE phải **đơn điệu** theo bin (tăng dần hoặc giảm dần). Notebook nói rõ: monotonic WoE giúp ổn định, dễ diễn giải, giảm overfit. Nếu không đơn điệu → gộp bin cho tới khi đơn điệu, trừ khi có lý do nghiệp vụ (ví dụ tuổi có quan hệ hình chữ U).

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.05)
tree.fit(df[[col]].fillna(-999), y)
edges = sorted(tree.tree_.threshold[tree.tree_.feature == 0])
bins = [-np.inf] + edges + [np.inf]
```

### B2. WoE và IV

Với quy ước **1 = good (event), 0 = bad (non-event)** như notebook LendingClub:

```
WoE_bin = ln( %non-event trong bin / %event trong bin )
IV      = Σ (%non-event − %event) × WoE
```

(Quy ước phổ biến hơn ngoài ngành là 1 = bad; khi đó `WoE = ln(%good/%bad)`. Dấu đổi, |IV| không đổi. **Ghi rõ quy ước vào code.**)

Ngưỡng IV chuẩn (notebook dùng đúng bảng này):

| IV | Ý nghĩa |
|---|---|
| < 0.02 | Không dự báo được → bỏ |
| 0.02 – 0.1 | Yếu |
| 0.1 – 0.3 | Trung bình |
| 0.3+ | Mạnh |
| > 0.5 | **Nghi ngờ** — overfit hoặc leakage |

Notebook LendingClub bỏ `application_type`, `initial_list_status`, `addr_state`, `purpose` vì IV < 0.02 — đây là cách feature selection rẻ và có căn cứ thống kê.

Cộng thêm công dụng: WoE cho biết **cách gộp category**. LendingClub nhìn bảng WoE thấy `ANY`, `MORTGAGE`, `NONE` có WoE gần nhau → gộp thành `other`.

### B3. Encode và train

Hai cách encode sau binning:
1. **Thay giá trị bằng WoE** — mỗi biến thành 1 cột số. Gọn, hệ số LR đọc trực tiếp.
2. **One-hot theo bin, drop first** — mỗi bin thành 1 cột dummy. Đây là cách notebook Scorecard làm; giữ được từng bin thành từng dòng trong scorecard cuối.

```python
model = LogisticRegression()      # cân nhắc C nhỏ / penalty='l1' để thưa hóa
model.fit(X, y)
summary = pd.DataFrame({'feature': X.columns, 'coef': model.coef_[0]})
```

Kiểm tra sau train: **dấu của mọi hệ số phải hợp trực giác nghiệp vụ**. Hệ số sai dấu = đa cộng tuyến hoặc binning sai, không phải "dữ liệu nó thế".

Bẫy đã ghi nhận trong notebook đó: cùng code, `sklearn 1.2.2` cho `intercept_ = -7.52` còn `1.5.1` cho `-3.49`. Bài học: **pin version** trong `requirements.txt`, và scorecard phải reproduce được.

### B4. Scaling ra điểm

Công thức chuẩn:
```
Score = Base + Factor × ln(odds),   Factor = PDO / ln(2)
```
Ví dụ: Base = 600 tại odds 50:1, PDO = 20 → mỗi 20 điểm thì odds gấp đôi.

Cách notebook Scorecard làm (scale tuyến tính hệ số về dải 300–900):
```python
min_sum = scorecard.groupby('feature_original')['coefficience'].min().sum()
max_sum = scorecard.groupby('feature_original')['coefficience'].max().sum()
scorecard['score_cal'] = scorecard['coefficience'] * (900-300) / (max_sum - min_sum)
```
Ý tưởng: hồ sơ xấu nhất có thể (chọn bin tệ nhất mọi biến) = 300, tốt nhất = 900. Kết quả cuối là bảng `(feature, bin, điểm)` cộng lại ra tổng điểm.

> Lưu ý kỹ thuật: đoạn convert ngược ra xác suất trong notebook đó (`1/(1+exp(-(score-300)/600))`) **không phải** nghịch đảo đúng của phép scale; nghịch đảo đúng là `PD = 1/(1+exp((score-Base)/Factor))`. Đọc để hiểu ý tưởng, đừng copy công thức đó.

### B5. Cutoff

Chọn ngưỡng điểm để duyệt/từ chối. Ba cách:

1. **Youden's J** — `J = TPR − FPR`, lấy ngưỡng cực đại J. Notebook dùng cách này. Thuần thống kê, coi hai loại lỗi ngang nhau → **không phù hợp tín dụng**.
2. **Theo lợi nhuận** — chọn ngưỡng cực đại `(số good được duyệt × lãi) − (số bad được duyệt × lỗ)`. Đây là cách đúng.
3. **Theo approval rate mục tiêu** — nghiệp vụ ấn định "duyệt 60% hồ sơ", lấy percentile 40 của điểm. Thực tế hay dùng nhất.

Sau khi có cutoff, chia dải điểm thành **risk grade** (A/B/C/D hoặc Low/Medium/High) để định giá lãi suất theo rủi ro, không chỉ duyệt/từ chối nhị phân.

## 5.3 Pipeline GBDT

```python
import lightgbm as lgb
params = dict(
    objective='binary', metric='auc',
    learning_rate=0.02, num_leaves=31,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    n_estimators=5000,
)
model = lgb.LGBMClassifier(**params)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(200)])
```

Ghi chú:
- Không cần scale, không cần impute, xử lý NaN native.
- `is_unbalance` / `scale_pos_weight`: **không cần** nếu chỉ quan tâm AUC (AUC bất biến với việc scale xác suất). Chỉ cần khi muốn xác suất đã calibrate hoặc dùng ngưỡng 0.5.
- Early stopping trên tập validation out-of-time.
- Feature importance: dùng `importance_type='gain'`, không dùng `'split'` mặc định. Tốt hơn nữa: SHAP.

Tham chiếu kết quả trên Home Credit Default Risk (notebook "Start Here"): LogisticRegression ≈ 0.67 → RandomForest ≈ 0.68 → LightGBM ≈ 0.75 public LB (điểm tốt nhất của notebook: **0.75262**). Top leaderboard cuộc thi ~0.80. Khoảng cách 0.75 → 0.80 gần như hoàn toàn đến từ **feature engineering trên các bảng phụ**, không từ tuning.

## 5.4 Validation

**Bắt buộc: out-of-time split.**

```
train : hồ sơ tháng 1–18
valid : tháng 19–21     (tune, early stopping)
test  : tháng 22–24     (chạm đúng 1 lần)
```

Random K-fold chỉ dùng để tune trong phạm vi tập train, và luôn phải kèm một đánh giá out-of-time. Home Credit Model Stability đẩy nguyên tắc này vào metric: tính gini cho **từng `WEEK_NUM`** rồi phạt xu hướng giảm.

Kiểm tra thêm:
- Đường gini theo tháng — dốc xuống là báo động.
- PSI giữa train và test — xem [06](06-metrics-validation-monitoring.md).
- Hiệu năng theo segment — mô hình có thể tốt tổng thể nhưng hỏng ở một segment.

## 5.5 Thứ tự làm việc đề xuất (khi có dữ liệu thật)

```
1. Định nghĩa label + performance window   → verify: bad rate ổn định theo tháng
2. Base table 1 dòng / 1 hồ sơ, chỉ depth 0 → verify: không có cột nào sinh sau T0
3. Baseline LightGBM trên depth 0          → verify: có số AUC out-of-time để so
4. Rà leakage (feature importance top 20)  → verify: từng feature giải thích được
5. Aggregate depth 1, depth 2              → verify: AUC tăng, không tăng thì bỏ
6. Binning + WoE + IV                      → verify: WoE đơn điệu, IV < 0.5
7. Logistic regression + scorecard          → verify: dấu hệ số hợp nghiệp vụ
8. Cutoff theo lợi nhuận / approval rate    → verify: có bảng P&L theo ngưỡng
9. PSI + gini theo tuần                     → verify: PSI < 0.1, gini không dốc xuống
```

Bước 3 trước bước 5 là có chủ ý: có baseline sớm để biết mỗi lớp dữ liệu thêm vào đáng giá bao nhiêu AUC.

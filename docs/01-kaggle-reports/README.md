# Report chi tiết từng link Kaggle

Mỗi file = 1 link trong [notes/task.txt](../../notes/task.txt). Cấu trúc thống nhất: nguồn → nội dung → điểm học được → cảnh báo → dùng lại được gì.

## Danh sách

| # | File | Link gốc | Loại |
|---|---|---|---|
| 1 | [01-nb-start-here-gentle-introduction.md](01-nb-start-here-gentle-introduction.md) | [willkoehrsen/start-here-a-gentle-introduction](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction) | Notebook |
| 2 | [02-nb-complete-eda-feature-importance.md](02-nb-complete-eda-feature-importance.md) | [codename007/home-credit-complete-eda-feature-importance](https://www.kaggle.com/code/codename007/home-credit-complete-eda-feature-importance) | Notebook |
| 3 | [03-nb-credit-risk-eda-defaults-segments-trends.md](03-nb-credit-risk-eda-defaults-segments-trends.md) | [beatafaron/credit-risk-eda-defaults-segments-trends-1](https://www.kaggle.com/code/beatafaron/credit-risk-eda-defaults-segments-trends-1) | Notebook |
| 4 | [04-nb-credit-risk-eda-woe-scorecard-2.md](04-nb-credit-risk-eda-woe-scorecard-2.md) | [beatafaron/credit-risk-eda-woe-scorecard-2](https://www.kaggle.com/code/beatafaron/credit-risk-eda-woe-scorecard-2) | Notebook |
| 5 | [05-comp-home-credit-model-stability.md](05-comp-home-credit-model-stability.md) | [home-credit-credit-risk-model-stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) | Cuộc thi |
| 6 | [06-comp-give-me-some-credit.md](06-comp-give-me-some-credit.md) | [GiveMeSomeCredit](https://www.kaggle.com/competitions/GiveMeSomeCredit) | Cuộc thi |
| 7 | [07-comp-home-credit-default-risk.md](07-comp-home-credit-default-risk.md) | [home-credit-default-risk](https://www.kaggle.com/competitions/home-credit-default-risk) | Cuộc thi |

## So sánh nhanh 3 bộ dữ liệu

| | GiveMeSomeCredit | Home Credit Default Risk | HC Model Stability |
|---|---|---|---|
| Năm | 2011 | 2018 | 2024 |
| Kích thước | 14.5 MB | 2.68 GB | ~26 GB (parquet/csv) |
| Số bảng | 1 | 7 (+3 phụ) | ~15 nhóm bảng, hàng chục file |
| Số cột | 11 | 346 | ~470 predictor |
| Train rows | 150,000 | 307,511 | ~1.5 triệu case |
| Target | `SeriousDlqin2yrs` (90+ DPD trong 2 năm) | `TARGET` (payment difficulties) | `target` (default sau một kỳ quan sát) |
| Metric | AUC | AUC | **gini stability metric** (custom) |
| Giải thưởng | $5,000 | $70,000 | $105,000 |
| Độ khó kỹ thuật | Thấp — tập tay | Trung bình — join + aggregate | Cao — bộ nhớ, tốc độ, stability |
| Dùng khi nào | Học quy trình, tuần này | Học feature engineering đa bảng | Học validation theo thời gian |

## So sánh nhanh 4 notebook

| | #1 Gentle Intro | #2 Complete EDA | #3 Defaults & Segments | #4 WoE & Scorecard |
|---|---|---|---|---|
| Dữ liệu | Home Credit Default Risk | Home Credit Default Risk | LendingClub 2014–18 | LendingClub (đã lọc từ #3) |
| Trọng tâm | Pipeline ML từ đầu tới cuối | EDA trực quan diện rộng | EDA + feature selection + so sánh 4 model | WoE/IV → LR → scorecard 300–900 |
| Mô hình | LR, RF, LightGBM | RandomForest (chỉ để lấy importance) | LR, RF, GBM, MLP | Logistic Regression |
| Giá trị chính | Xử lý anomaly, encoding, domain features | Mẫu biểu đồ tỷ lệ theo category | Quy trình chọn feature | Quy trình dựng scorecard |
| Rủi ro nội dung | Thấp — chuẩn mực | Thấp — thuần mô tả | **Cao — leakage nặng** | **Cao — kế thừa leakage, vài công thức sai** |

## Lộ trình đọc đề xuất

```
06 (GiveMeSomeCredit — hiểu bài toán tối giản)
 → 07 (Home Credit Default Risk — hiểu cấu trúc dữ liệu thật)
 → 01 (Gentle Intro — pipeline chuẩn)
 → 02 (Complete EDA — kỹ thuật trực quan)
 → 03 + 04 (LendingClub — WoE/scorecard, đọc kèm phần cảnh báo)
 → 05 (Model Stability — validation theo thời gian)
```

## Ghi chú về nguồn

Trang Kaggle render bằng JavaScript nên fetch trực tiếp chỉ ra tiêu đề. Nội dung trong các report này lấy bằng:
- **Trang cuộc thi** (#5, #6, #7): fetch qua reader proxy → lấy được nguyên văn Overview / Evaluation / Data description. Số liệu trong report là nguyên văn từ trang.
- **Notebook #3, #4**: tác giả link notebook sang GitHub ([BeataFaron/credit-risk-psi-scorecards](https://github.com/BeataFaron/credit-risk-psi-scorecards)) → tải được file `.ipynb` gốc, trích toàn bộ markdown + code cell. Code và con số trích dẫn là nguyên văn.
- **Notebook #1, #2**: chỉ lấy được **mục lục đầy đủ** và metadata (điểm LB, runtime). Phần mô tả nội dung dựa trên mục lục cộng kiến thức về bộ dữ liệu Home Credit Default Risk — chỗ nào là suy luận đều được đánh dấu trong file.

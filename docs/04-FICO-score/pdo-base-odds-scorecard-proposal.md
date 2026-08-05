# Đề xuất scorecard theo base odds và PDO

> **Trạng thái: Planned — chưa có trong runtime hiện tại.** Báo cáo này đặc tả một
> phương án thay thế hoặc chạy song song với min–max scaling. Báo cáo về hành vi
> đang hoạt động nằm tại [Cách tính điểm hiện tại](./scoring_method.md).

## 1. Kết luận ngắn

Đề xuất giữ nguyên pipeline train-only cho binning, WoE và Logistic Regression,
nhưng thay phép ép biên 300–850 bằng scale theo `base_score=600`,
`base_odds=50:1` và `PDO=20`. Intercept được đưa trực tiếp vào `base_points`;
score float có thể đổi ngược về log-odds và xác suất. Đây là cải thiện tiềm năng về
khả năng giải thích, audit và versioning, không phải bằng chứng rằng AUC sẽ tăng và
không tạo ra FICO Score chính thức.


> **Planned — chưa có trong runtime hiện tại:** phần này đặc tả implementation nên
> bổ sung để thay thế hoặc chạy song song với min–max scaling. Các công thức và code
> dưới đây chưa phải bằng chứng rằng artifact hiện tại đã dùng PDO.

## 2. Why — min–max 300–850 chưa biểu diễn ý nghĩa của một điểm

Implementation hiện tại bảo đảm tổ hợp bin thấp nhất bằng 300 và tổ hợp cao nhất
bằng 850. Cách này dễ trình bày nhưng có bốn hạn chế:

1. **Một điểm không có ý nghĩa odds cố định.** Không thể nói tăng 20 điểm tương ứng
   odds good:bad tăng gấp đôi.
2. **Intercept bị tách khỏi score.** Hai output `predict_proba()` và integer score
   cùng dùng coefficient nhưng không có phép đổi chính xác qua lại.
3. **Scale thay đổi khi model thay đổi.** Chỉ cần thêm feature hoặc thay một extreme
   bin là `C_min`, `C_max` và điểm của mọi bin có thể đổi, ngay cả khi phần lớn quan
   hệ rủi ro gần như giữ nguyên.
4. **Dải 300–850 dễ bị đọc nhầm là FICO Score.** Dải giống nhau không tạo ra cùng
   model, dữ liệu, calibration hay ý nghĩa nghiệp vụ.

Tiêu chí thành công của phương án mới là:

- score dùng đầy đủ intercept và contribution của Logistic Regression;
- `base_score=600` tương ứng đúng `base_odds=50:1` theo convention good:bad;
- mỗi lần good:bad odds tăng gấp đôi, score tăng đúng `PDO=20` trước rounding;
- score cao luôn tương ứng `PD_bad` thấp hơn;
- có thể khôi phục log-odds và PD từ score chưa làm tròn;
- không làm thay đổi binning, WoE, feature selection hay protocol train-only hiện có.

## 3. Chốt convention trước khi viết công thức

Đây là phần dễ gây lỗi dấu nhất. Đề xuất này cố định contract:

- `TARGET=1` là bad;
- Logistic Regression học
  \(z=logit(PD_{bad})=\ln(P_{bad}/P_{good})\);
- `base_odds` được định nghĩa là **good:bad**, không phải bad:good;
- score cao hơn nghĩa là rủi ro thấp hơn;
- WoE hiện tại vẫn là \(\ln(\%Good/\%Bad)\).

Với `base_odds=50`, điểm chuẩn xảy ra tại:

$$
\frac{P_{good}}{P_{bad}}=50
\quad\Longleftrightarrow\quad
z=\ln\left(\frac{P_{bad}}{P_{good}}\right)=-\ln(50)
$$

Đặt:

$$
B=\frac{PDO}{\ln 2}
$$

$$
A=base\_score-B\ln(base\_odds)
$$

thì công thức score là:

$$
Score=A-Bz
$$

Thay \(z=\beta_0+\sum_j\beta_j WoE_j\):

$$
Score=
\underbrace{A-B\beta_0}_{BasePoints}
+
\sum_j\underbrace{(-B\beta_j WoE_j)}_{BinPoints_j}
$$

Với `PDO=20`, `base_score=600`, `base_odds=50`:

$$
B=\frac{20}{\ln2}\approx28,8539
$$

$$
A=600-28,8539\ln(50)\approx487,1229
$$

Kiểm tra bất biến PDO: nếu good:bad odds tăng gấp đôi thì bad log-odds giảm
`ln(2)` và:

$$
Score_{new}=A-B(z-\ln2)=Score_{old}+PDO
$$

Một số tài liệu viết `A = base_score + B × ln(base_odds)`. Công thức đó chỉ có thể
đúng khi định nghĩa odds và chiều score đi kèm cũng được đổi nhất quán. Trộn công
thức `A` của bad:good odds với `base_odds=good:bad` là lỗi dấu.

## 4. Vì sao đoạn code đề xuất ban đầu cần sửa

Đoạn code ban đầu đã đọc:

```python
lr_intercept = lr.intercept_[0]
```

nhưng không đưa `lr_intercept` vào điểm. Nó còn dùng đồng thời:

```python
A = base_score + B * np.log(base_odds)
feature_score = coef * woe_value * B + A / n_features
```

Hai dấu cộng này không khớp contract `TARGET=1 là bad`, `base_odds=good:bad` và
`score cao là tốt`. Kết quả không bảo đảm 600 điểm tại odds 50:1 và cũng không bảo
đảm tăng đúng 20 điểm khi odds tốt tăng gấp đôi.

Thay vì chia hằng số vào mọi feature, implementation nên lưu `base_points` riêng.
Cách này làm provenance của intercept rõ ràng và tránh phải sửa điểm của mọi feature
khi số feature thay đổi.

## 5. Implementation Python đề xuất

Hàm dưới đây chỉ thay phần scale sau khi binning, WoE và LR đã được fit. Nó giữ
`points_float` để bảo toàn quan hệ odds; chỉ round **tổng điểm cuối** khi cần hiển
thị.

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PDOScale:
    base_score: float = 600.0
    base_odds_good_to_bad: float = 50.0
    pdo: float = 20.0

    def __post_init__(self) -> None:
        if self.base_odds_good_to_bad <= 0:
            raise ValueError("base_odds_good_to_bad must be positive")
        if self.pdo <= 0:
            raise ValueError("pdo must be positive")

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(
            self.base_odds_good_to_bad
        )


def scorecard_from_lr_pdo(
    woe_frame: pd.DataFrame,
    target: pd.Series,
    woe_tables: dict[str, pd.DataFrame],
    *,
    scale: PDOScale = PDOScale(),
) -> tuple[LogisticRegression, pd.DataFrame, dict[str, float]]:
    """Fit bad=1 LR and create a good:bad odds/PDO scorecard."""
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(woe_frame, target)

    coefficients = dict(
        zip(woe_frame.columns, model.coef_[0], strict=True)
    )
    base_points = scale.offset - scale.factor * float(model.intercept_[0])

    rows: list[pd.DataFrame] = []
    for feature, table in woe_tables.items():
        feature_table = table.copy()
        coefficient = float(coefficients[feature])
        feature_table["coefficient"] = coefficient
        feature_table["log_odds_contribution"] = (
            coefficient * feature_table["woe"]
        )
        feature_table["points_float"] = (
            -scale.factor * feature_table["log_odds_contribution"]
        )
        rows.append(feature_table)

    scorecard = pd.concat(rows, ignore_index=True)
    metadata = {
        "target_event": 1.0,
        "base_score": scale.base_score,
        "base_odds_good_to_bad": scale.base_odds_good_to_bad,
        "pdo": scale.pdo,
        "factor": scale.factor,
        "offset": scale.offset,
        "model_intercept": float(model.intercept_[0]),
        "base_points": base_points,
    }
    return model, scorecard, metadata


def total_score(
    mapped_bin_points: pd.DataFrame,
    *,
    base_points: float,
    rounded: bool = True,
) -> np.ndarray:
    score = base_points + mapped_bin_points.sum(axis=1).to_numpy()
    return np.rint(score).astype(int) if rounded else score
```

Trong pipeline thật, `mapped_bin_points` phải được tạo bằng chính `bin_edges` và
mapping `feature, bin -> points_float` đã fit trên train. Không được tính lại bin
hoặc WoE trên validation, test hay competition data.

Nên lưu tối thiểu:

- `scorecard_pdo.csv`: feature, bin, WoE, coefficient, contribution và
  `points_float`;
- `scorecard_pdo_metadata.json`: target convention, odds convention, base score,
  base odds, PDO, factor, offset, intercept và base points;
- model, bin edges và WoE tables như artifact hiện tại.

## 6. Quan hệ score ↔ odds ↔ PD

Với score float chưa round:

$$
z=\frac{A-Score}{B}
$$

$$
PD_{bad}=sigmoid(z)=\frac{1}{1+e^{-z}}
$$

$$
Odds_{good:bad}=e^{-z}
=\exp\left(\frac{Score-A}{B}\right)
$$

Đây là lợi ích lớn so với min–max scaling: score, odds và PD cùng nằm trên một trục
toán học có thể audit. Sau khi round score về integer, PD đảo ngược chỉ là xấp xỉ;
`model.predict_proba()` vẫn là nguồn PD chính xác.

## 7. Potential benefits

| Vấn đề hiện tại                         | Cơ chế của PDO scaling                            | Lợi ích tiềm năng                                         |
| -------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| Một điểm không có ý nghĩa cố định  | `B = PDO / ln(2)`                                  | Giải thích được thay đổi score bằng thay đổi odds   |
| Intercept không có trong score             | `BasePoints = A - B × intercept`                  | Score và decision function truy vết cùng một model        |
| Khó đổi score về PD                      | `z = (A - Score) / B`                              | Có phép đối chiếu score–odds–PD rõ ràng              |
| Min/max đổi khi extreme bin đổi          | Scale cố định bằng base score, base odds và PDO | Score có khả năng so sánh qua các lần retrain tốt hơn |
| Chia base vào feature che khuất provenance | Lưu base points riêng                              | Audit coefficient/bin contribution dễ hơn                   |
| Làm tròn từng bin tích lũy sai số      | Giữ float bin points, round tổng cuối             | Giảm sai lệch giữa score và log-odds                      |

“Có khả năng so sánh” không có nghĩa score tự động comparable giữa HCDR, HCMS và
GiveMeSomeCredit. Muốn so trực tiếp, các model còn phải có cùng target semantics,
performance window và calibration. Ba competition hiện không thỏa contract chung
đó.

## 8. Trade-off và giới hạn

1. **Không còn bảo đảm nằm trong 300–850.** PDO scaling ưu tiên ý nghĩa odds. Có thể
   tạo `display_score = clip(raw_score, 300, 850)`, nhưng phải giữ `raw_score` làm
   nguồn audit vì clipping phá quan hệ nghịch đảo ở hai biên.
2. **Base odds là tham số policy.** `50:1` không được suy ra từ FICO booklet và không
   mặc nhiên đúng cho population HCDR. Giá trị này cần được phê duyệt và version hóa.
3. **Calibration quyết định ý nghĩa xác suất.** Logistic Regression có log-odds nội
   tại nhưng PD vẫn cần calibration test; AUC tốt không chứng minh PD đúng.
4. **Rounding làm mất tính nghịch đảo tuyệt đối.** Chỉ score float khôi phục chính
   xác decision function.
5. **Không biến score thành FICO Score.** PDO là kỹ thuật scale scorecard phổ biến;
   nó không cung cấp model, credit-bureau data hay calibration độc quyền của FICO.
6. **Không sửa các giới hạn validation khác.** Random split, cutoff chọn trên test,
   fairness, reject inference và monitoring thời gian vẫn cần xử lý riêng.

## 9. Kế hoạch kiểm chứng trước khi thay implementation hiện tại

Nên chạy min–max và PDO song song trước khi migration. Bộ test tối thiểu:

1. **Base point invariant:** `z = -ln(50)` phải cho score float bằng 600.
2. **PDO invariant:** thay `z` bằng `z - ln(2)` phải tăng đúng 20 điểm.
3. **Direction invariant:** với hai hồ sơ, `PD_bad` thấp hơn phải có score cao hơn.
4. **Reconstruction:** tổng `base_points + bin_points` phải bằng
   `A - B × model.decision_function(X_woe)` trong tolerance số thực.
5. **Artifact round-trip:** load model, bins, WoE table và metadata phải tái tạo cùng
   score theo cùng thứ tự hồ sơ.
6. **No leakage:** bin, WoE, coefficient và scale metadata chỉ được fit/derive từ
   train; validation/test chỉ transform.
7. **Rounding audit:** đo chênh lệch PD giữa score float, integer score và
   `predict_proba()`.
8. **Policy comparison:** đóng băng cutoff trên validation rồi so approval rate,
   bad rate và expected cost trên test.

Chỉ sau khi các invariant trên qua test mới nên đổi cutoff/report chính sang PDO.
Trong giai đoạn chuyển tiếp, artifact phải ghi rõ `scaling_method=minmax_300_850`
hoặc `scaling_method=pdo_odds` để tránh trộn hai loại điểm.


## 10. Nguồn và phạm vi bằng chứng

- **Verified current code:** [scorecard scaling dùng chung](../../src/credit_scoring/scorecard.py)
  và [pipeline HCDR](../../src/home_credit_default_rate/pipeline.py).
- **Verified current behavior:** [báo cáo scorecard HCDR hiện tại](./scoring_method.md).
- **Nguồn phân biệt với FICO chính thức:** [Understanding FICO Scores](./Understanding_FICO_Scores_5181BK.pdf).
- **Planned:** toàn bộ API `PDOScale`, `scorecard_from_lr_pdo()`, artifact PDO và
  migration tests trong báo cáo này chưa được triển khai vào `src/`.

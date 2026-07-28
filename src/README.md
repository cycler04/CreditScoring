# Source code

`src/credit_scoring/` chứa logic Python tái sử dụng. Các command dành cho người
dùng nằm trong [`scripts/`](../scripts/README.md); artefact không được ghi vào
`src/`.

## Import package

Chạy từ thư mục gốc dự án:

```bash
uv sync --locked
uv run python
```

Sau đó có thể import các API công khai:

```python
from credit_scoring import (
    bin_by_tree,
    gini_by_period,
    psi,
    scorecard_from_lr,
    woe_iv,
)
```

## Các module

### `data.py`

- `find_training_csv(raw_dir)`: tìm `cs-training.csv`.
- `load_training_data(csv_path)`: đọc CSV, bỏ cột số thứ tự Kaggle và kiểm tra
  target `SeriousDlqin2yrs`.
- `clean_features(df)`: xử lý `age=0`, mã delinquency `96/98`, tạo anomaly
  flags và trả về `(clean_df, findings_df)`.

Ví dụ:

```python
from pathlib import Path
from credit_scoring.data import clean_features, load_training_data

raw = load_training_data(Path("datasets/raw/cs-training.csv"))
clean, findings = clean_features(raw)
```

### `eda.py`

- `bad_rate_by_decile(df, feature_columns, target)`: bad rate theo quantile.
- `write_eda_outputs(...)`: sinh bảng CSV và plots vào thư mục được truyền vào.

EDA của pipeline chỉ chạy trên training split, không nhìn validation/test.

### `metrics.py`

`psi()` trả về cả tổng PSI và bảng contribution để audit:

```python
from credit_scoring import psi

psi_value, detail = psi(expected_scores, actual_scores, bins=10)
```

`gini_by_period()` cần một cột thời gian/kỳ:

```python
from credit_scoring import gini_by_period

monthly = gini_by_period(
    frame,
    period_col="month",
    target_col="default",
    score_col="predicted_pd",
)
```

GiveMeSomeCredit không có cột thời gian nên hàm này được unit test nhưng không
được dùng để tạo kết quả theo kỳ cho dataset hiện tại.

### `scorecard.py`

- `bin_by_tree()`: tìm bin edges bằng decision tree, mặc định tối đa 6 leaf.
- `woe_iv()`: quy ước target `1=bad`, `WoE=ln(%good/%bad)`.
- `scorecard_from_lr()`: fit Logistic Regression trên WoE và scale thành điểm
  integer 300–850; điểm cao nghĩa là rủi ro thấp.

Ví dụ WoE/IV:

```python
from credit_scoring import bin_by_tree, woe_iv

edges = bin_by_tree(train["age"], train["SeriousDlqin2yrs"])
table, iv = woe_iv(
    train,
    col="age",
    target="SeriousDlqin2yrs",
    bins=edges,
)
```

Pipeline tự giảm tree depth cho tới khi WoE của các bin số đơn điệu. Missing là
một bin riêng.

### `pipeline.py`

`run_pipeline(project_root)` chạy:

1. Load và làm sạch dữ liệu.
2. Stratified split 60/20/20.
3. EDA trên training split.
4. Logistic Regression thô và LightGBM.
5. WoE/IV, Logistic Regression trên WoE và scorecard.
6. Approval cutoffs và score PSI.

Nên gọi qua `scripts/run_pipeline.py` thay vì gọi trực tiếp.

## Kiểm thử

```bash
./scripts/check.sh
```

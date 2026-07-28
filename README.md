# CreditScoring

Pipeline thực hành GiveMeSomeCredit theo
[checklist tuần](docs/00-tong-quan/07-ke-hoach-tuan-checklist.md).

## Cấu trúc

- `src/`: logic Python tái sử dụng và inspection dữ liệu.
- `scripts/`: command tải dữ liệu và chạy pipeline.
- `datasets/`: dữ liệu raw/processed, không commit.
- `outputs/`: EDA, metrics, models và scorecard, không commit.
- `notebooks/`: source và metadata của các Kaggle notebook tham khảo.
- `docs/`: kiến thức và playbook.

## Chạy

```bash
uv sync --locked
./scripts/run_all.sh
```

Hướng dẫn chi tiết:

- [Sử dụng Python API trong `src/`](src/README.md).
- [Các command trong `scripts/`](scripts/README.md).
- [Danh sách Kaggle notebooks local](notebooks/README.md).

`.env` cần chứa `KAGGLE_API_TOKEN`. Script không ghi token ra log hay file khác.
Python 3.14 được khóa trong `.python-version`; dependency đầy đủ được khóa trong
`uv.lock`.

Kết quả chính:

- `outputs/eda/`: target, missing, anomaly, bad rate decile và plots.
- `outputs/models/metrics.csv`: AUC, Gini, KS của LR, LightGBM, LR-WoE.
- `outputs/scorecard/`: IV, WoE, bins, scorecard 300–850, cutoff và PSI.
- `outputs/run_summary.json`: tóm tắt lần chạy.

GiveMeSomeCredit không có cột thời gian. Pipeline dùng stratified random split
60/20/20 và ghi rõ giới hạn này; không giả vờ đó là out-of-time validation.

Kiểm tra code và dependency:

```bash
./scripts/check.sh
```

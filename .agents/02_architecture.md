# Kiến trúc

## Luồng chính

```text
Kaggle/public mirror
  -> datasets/raw/cs-training.csv
  -> load + clean
  -> stratified split 60/20/20
  -> EDA
  -> LR | LightGBM | LR-WoE
  -> AUC/Gini/KS + scorecard/cutoff/PSI
  -> outputs/
```

## Thành phần

- `src/credit_scoring/data.py`: tìm, đọc và làm sạch dữ liệu.
- `src/credit_scoring/eda.py`: bảng và biểu đồ EDA.
- `src/credit_scoring/metrics.py`: PSI và Gini theo period.
- `src/credit_scoring/scorecard.py`: binning, WoE/IV và scorecard.
- `src/credit_scoring/pipeline.py`: orchestration, split, model và artifact.
- `scripts/download_data.py`: tải dataset mà không làm lộ token.
- `scripts/run_pipeline.py`, `scripts/run_all.sh`: entry point chạy pipeline.
- `scripts/check.sh`: kiểm tra code và test.
- `tests/`: unit test.

## Dữ liệu và artifact

- `datasets/raw/`: input tải về, không track.
- `datasets/processed/`: dữ liệu trung gian nếu phát sinh, không track.
- `outputs/eda/`: bảng và plot EDA.
- `outputs/models/`: metric và model artifact.
- `outputs/scorecard/`: IV, WoE, bins, scorecard, cutoff và PSI.
- `outputs/run_summary.json`: tóm tắt lần chạy.

`docs/` là knowledge/playbook; `notebooks/` là nguồn tham khảo. Không mặc định coi code
trong notebook là implementation chuẩn của package.

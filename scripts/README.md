# Scripts

Các script đã được nhóm theo mục đích dưới các thư mục:

- `data/`: tải và chuẩn bị dữ liệu
- `pipelines/`: chạy pipeline, EDA và phân tích mô hình
- `notebooks/`: helper cho notebook và submission artifacts
- `ops/`: script vận hành và wrapper

Các script phải được chạy từ thư mục gốc dự án. Dữ liệu được ghi vào
`datasets/`, còn báo cáo/model được ghi vào `outputs/`.

## Chuẩn bị môi trường

```bash
uv sync --locked
```

`uv` đọc `.python-version`, tạo `.venv`, cài package `credit_scoring` từ
`src/` và dùng đúng dependency trong `uv.lock`.

Tạo `.env`:

```dotenv
KAGGLE_API_TOKEN=your_kaggle_token
```

Tài khoản Kaggle phải chấp nhận rules của cuộc thi GiveMeSomeCredit. Token không
được in ra log và `.env` đã nằm trong `.gitignore`.

## Chạy toàn bộ

```bash
./scripts/ops/run_all.sh
```

Command này lần lượt:

1. Kiểm tra/tải dữ liệu bằng `download_data.py`.
2. Chạy EDA, models, WoE/IV, scorecard và PSI bằng `run_pipeline.py`.

Nếu đủ 4 file trong `datasets/raw/give-me-some-credit/`, bước download được bỏ qua.

## Tải lại dữ liệu chính thức

Tải khi chưa có dữ liệu:

```bash
uv run python scripts/data/download_data.py
```

Ép tải lại trực tiếp từ competition:

```bash
uv run python scripts/download_data.py --force
```

Nguồn và SHA-256 của bốn file được lưu tại `datasets/raw/give-me-some-credit/source.json`. Nếu
competition trả 403, script mới fallback sang public Kaggle mirror và ghi rõ
nguồn này trong metadata.

## Chỉ chạy pipeline

Khi `datasets/raw/give-me-some-credit/cs-training.csv` đã có:

```bash
uv run python scripts/pipelines/run_pipeline.py
```

Output chính:

| Đường dẫn | Nội dung |
|---|---|
| `datasets/processed/` | Dữ liệu clean và split membership |
| `outputs/eda/` | Target, missing, anomaly, decile, plots và notebook |
| `outputs/models/metrics/metrics.csv` | AUC, Gini, KS |
| `outputs/models/metrics/roc_auc_curve.png` | ROC-AUC trên test split của ba mô hình |
| `outputs/models/metrics/gini_curve.png` | Cumulative gains và Gini trên test split |
| `outputs/models/metrics/ks_curve.png` | KS separation trên test split |
| `outputs/models/*.joblib` | Logistic Regression và LightGBM |
| `outputs/scorecard/iv_summary.csv` | IV và trạng thái monotonic |
| `outputs/scorecard/scorecard.csv` | Bảng điểm 300–850 |
| `outputs/scorecard/approval_cutoffs.csv` | Cutoff ở approval 60/70/80% |
| `outputs/scorecard/score_psi_detail.csv` | Chi tiết score PSI |
| `outputs/run_summary.json` | Tóm tắt lần chạy |

GiveMeSomeCredit không có cột thời gian. Pipeline dùng stratified random split
60/20/20; kết quả không phải out-of-time validation.

## Tải Kaggle notebooks tham khảo

```bash
./scripts/ops/download_notebooks.sh
```

Command tải source và metadata của cả 4 notebook trong `notes/task.txt` vào
`notebooks/`. Chi tiết xem [`notebooks/README.md`](../notebooks/README.md).

Tải lại public code tìm được từ các team top leaderboard:

```bash
./scripts/ops/download_leaderboard_notebooks.sh
```

Trạng thái team nào có/không có public code được ghi tại
[`notebooks/leaderboard/README.md`](../notebooks/leaderboard/README.md).

Tải lại top 10 notebook theo vote từ Code tab GiveMeSomeCredit:

```bash
./scripts/ops/download_top_voted_givemesomecredit.sh
```

Snapshot và validation status nằm tại
[`notebooks/top-voted/GiveMeSomeCredit/`](../notebooks/top-voted/GiveMeSomeCredit/README.md).

Tải lại top 10 theo vote cho Home Credit Default Risk và Model Stability:

```bash
./scripts/ops/download_top_voted_other_competitions.sh
```

## Đồng bộ artifact lên tho2

```bash
./scripts/ops/push_to_tho2.sh
```

Script copy `docs/`, `notebooks/`, `datasets/` và `outputs/` tới
`vinrobotics:~/Dung_Workspace/testing/`. Transfer có thể tiếp tục sau khi gián
đoạn và script kiểm tra số file cùng tổng số byte của từng thư mục sau khi copy.
Script không xóa file chỉ có ở remote.

Kéo các thư mục đó từ tho2 về local:

```bash
./scripts/ops/pull_from_tho2.sh
```

Script cập nhật local từ `vinrobotics:~/Dung_Workspace/testing/`, tiếp tục được
transfer gián đoạn và không xóa file chỉ có ở local.

## Kiểm tra

```bash
./scripts/ops/check.sh
```

Script chạy unit tests, `pip check`, kiểm tra shell syntax và compile Python.

## Sinh lại biểu đồ Home Credit từ artifact hiện có

Không cần huấn luyện lại model khi các CSV trong `outputs/hcdr/` và
`outputs/hcms/` đã tồn tại:

```bash
uv run python scripts/pipelines/generate_hc_diagrams.py --dataset all
```

Có thể thay `all` bằng `hcdr` hoặc `hcms`. Pipeline đầy đủ cũng tự sinh ROC, Gini,
KS, feature importance và các biểu đồ EDA/stability vào cây `outputs/` tương ứng.
Để backfill cả ROC/Gini/KS từ model và processed holdout đã lưu:

```bash
uv run python scripts/pipelines/generate_hc_diagrams.py \
  --dataset all --with-predictions
```

## EDA chi tiết HCDR từ notebook tham khảo

Tái lập các phần missing data, phân phối biến, bad rate theo nhóm, tương quan
Pearson và Random Forest feature importance của notebook top-voted HCDR thành
CSV/PNG tĩnh:

```bash
uv run python scripts/pipelines/run_hcdr_notebook_eda.py
```

Artifact và provenance được ghi tại `outputs/hcdr/eda/`. Smoke test nhanh, không
fit Random Forest:

```bash
uv run python scripts/pipelines/run_hcdr_notebook_eda.py \
  --sample-rows 1000 --skip-feature-importance \
  --output-dir /tmp/hcdr-eda-smoke
```

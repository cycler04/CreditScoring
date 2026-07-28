# Kaggle notebooks

Thư mục này chứa source của 4 Kaggle notebook được liệt kê trong
[`notes/task.txt`](../notes/task.txt). Mỗi notebook nằm trong một thư mục riêng
và đi kèm `kernel-metadata.json` gốc để giữ thông tin nguồn, dataset và Kaggle
runtime.

| # | Notebook local | Kaggle source | Dữ liệu chính |
|---|---|---|---|
| 1 | [`start-here-a-gentle-introduction.ipynb`](01-start-here-a-gentle-introduction/start-here-a-gentle-introduction.ipynb) | `willkoehrsen/start-here-a-gentle-introduction` | Home Credit Default Risk |
| 2 | [`home-credit-complete-eda-feature-importance.ipynb`](02-home-credit-complete-eda-feature-importance/home-credit-complete-eda-feature-importance.ipynb) | `codename007/home-credit-complete-eda-feature-importance` | Home Credit Default Risk |
| 3 | [`credit-risk-eda-defaults-segments-trends-1.ipynb`](03-credit-risk-eda-defaults-segments-trends/credit-risk-eda-defaults-segments-trends-1.ipynb) | `beatafaron/credit-risk-eda-defaults-segments-trends-1` | Lending Club |
| 4 | [`credit-risk-eda-woe-scorecard-2.ipynb`](04-credit-risk-eda-woe-scorecard/credit-risk-eda-woe-scorecard-2.ipynb) | `beatafaron/credit-risk-eda-woe-scorecard-2` | Lending Club |

## Tải mới/cập nhật

Từ thư mục gốc dự án:

```bash
./scripts/download_notebooks.sh
```

Script dùng Kaggle token trong environment thông qua Kaggle CLI và kéo source
version hiện tại cùng metadata.

## Chạy notebook

Các file là source Kaggle nguyên bản, không được sửa đường dẫn input để giả vờ
rằng chúng chạy trực tiếp với GiveMeSomeCredit:

- Notebook 1–2 tham chiếu dataset của competition Home Credit Default Risk.
- Notebook 3–4 tham chiếu nhiều Lending Club datasets và Kaggle utility
  notebooks, được liệt kê chính xác trong `kernel-metadata.json`.
- Docker image/runtime gốc cũng nằm trong metadata. Môi trường uv của project
  chỉ đảm bảo pipeline trong `src/`; nó không cam kết tái tạo toàn bộ Kaggle
  runtime cũ của bốn notebook.

Mở notebook local bằng editor hỗ trợ Jupyter, hoặc cài Jupyter riêng nếu cần.

Public code tìm được từ các team top leaderboard được lưu riêng tại
[`leaderboard/`](leaderboard/README.md). Các artefact ở đó chủ yếu là Kaggle
script kernels, không phải notebook `.ipynb` và không được coi mặc định là final
winning submission.

Top 10 notebook theo vote của Code tab GiveMeSomeCredit được lưu tại
[`top-voted/GiveMeSomeCredit/`](top-voted/GiveMeSomeCredit/README.md). Đây là
community notebooks xuất bản sau cuộc thi, không phải code của top leaderboard
teams năm 2011.

Hai Code-tab snapshot còn lại:

- [`top-voted/home-credit-default-risk/`](top-voted/home-credit-default-risk/README.md)
- [`top-voted/home-credit-model-stability/`](top-voted/home-credit-model-stability/README.md)

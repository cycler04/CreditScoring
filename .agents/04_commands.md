# Lệnh

Chạy từ repository root.

## Khảo sát an toàn

```bash
git status --short
git log -5 --oneline --decorate
uv --version
python --version
```

## Môi trường

```bash
uv sync --locked
```

Python 3.14 nằm trong `.python-version`; không tự nới version hoặc cập nhật lockfile nếu
tác vụ không yêu cầu.

## Kiểm tra

```bash
./scripts/check.sh
uv run python -m unittest discover -s tests -v
```

## Chạy pipeline

```bash
./scripts/run_all.sh
```

Lệnh full có thể tải dữ liệu và cần `KAGGLE_API_TOKEN` trong `.env`. Không chạy download
hoặc pipeline full khi chỉ cần unit test. Có thể chạy entry point Python trực tiếp:

```bash
uv run python scripts/run_pipeline.py
```

## Hạ tầng agent

```bash
uv run python .agents/scripts/01_validate_workspace.py --full
```

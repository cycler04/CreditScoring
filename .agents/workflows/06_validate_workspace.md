# Validate workspace

## Khi dùng

Dùng sau khi sửa `.agents/`, `.codex/`, `AGENTS.md`, dependency hoặc entry point.

## Quy trình

1. Xem `git status --short` và bảo toàn thay đổi đang dở.
2. Chạy validator hạ tầng:

   ```bash
   uv run python .agents/scripts/01_validate_workspace.py --full
   ```

3. Chạy kiểm tra project:

   ```bash
   ./scripts/check.sh
   ```

4. Nếu thay đổi pipeline, bắt đầu bằng unit test hoặc sample nhỏ. Chỉ chạy
   `./scripts/run_all.sh` khi dataset, token và thời gian chạy nằm trong phạm vi tác vụ.
5. Báo cáo lệnh đã chạy, kết quả, phần chưa chạy và lý do.

## Tiêu chí hoàn tất

- Không có link nội bộ hỏng, numbering lỗi hoặc TOML/YAML không hợp lệ.
- Agent infrastructure không bị `.gitignore` bỏ qua.
- Unit test và check liên quan pass.
- Không có secret, dataset hay output sinh ra bị đưa vào Git.

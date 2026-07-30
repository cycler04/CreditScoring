# Quy ước

## Knowledge và tài liệu

- `.agents/` là nguồn duy nhất cho knowledge, memory, plan, workflow và skill của agent.
- Một sự thật chỉ có một nơi sở hữu; file khác liên kết tới nơi đó thay vì sao chép.
- Tài liệu viết tiếng Việt rõ ràng; code, identifier và docstring dùng tiếng Anh.
- Báo cáo phải tách nguồn/dữ kiện, diễn giải, giả thuyết, giới hạn và câu hỏi mở.
- Không ghi vào memory trạng thái có thể đọc trực tiếp từ code hoặc Git.

## Code và mô hình

- Tuân theo Python 3.14 và dependency trong `uv.lock`; ưu tiên `uv run`.
- Dùng type hints, `pathlib.Path`, `snake_case` cho hàm/biến và `PascalCase` cho class.
- Không gây data leakage giữa train, validation và test.
- Ghi rõ target, split, random seed, feature set và metric khi thay đổi mô hình.
- Scorecard/WoE phải giữ provenance của bins, smoothing và mapping.
- Thay đổi nhỏ phải có test tương ứng; không sửa notebook vendor để ép theo style package.

## Output và an toàn

- Không commit dataset, output sinh ra, model artifact, `.env` hoặc secret.
- Không log token Kaggle hay credential.
- Trước khi xóa, resolve đường dẫn và xác nhận nó nằm dưới workspace.
- Không commit, push hoặc thay đổi hệ thống bên ngoài nếu chưa được yêu cầu rõ.
- Bảo toàn thay đổi đang dở; không dùng reset/checkout để dọn worktree.

## Ngoại lệ do tool

`AGENTS.md`, `SKILL.md`, `MEMORY.md`, `agents/openai.yaml` và
`.codex/config.toml` giữ tên cố định theo cơ chế của tool.

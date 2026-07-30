# Hướng dẫn cho AI agent

`.agents/` là nguồn sự thật duy nhất cho knowledge, plan, memory, workflow và skill của
workspace này. Không tạo bản sao project-specific trong `~/.codex/`, `.codex/` hoặc
`.claude/`.

Đọc theo thứ tự trước khi làm việc:

1. [Tổng quan](.agents/01_overview.md)
2. [Kiến trúc](.agents/02_architecture.md)
3. [Quy ước](.agents/03_conventions.md)
4. [Lệnh](.agents/04_commands.md)

Sau đó đọc workflow phù hợp trong [`.agents/workflows/`](.agents/workflows/) và skill
tương ứng trong [`.agents/skills/`](.agents/skills/). Bối cảnh dài hạn nằm tại
[`.agents/memory/MEMORY.md`](.agents/memory/MEMORY.md); kế hoạch hiện hành nằm trong
[`.agents/plans/`](.agents/plans/).

Với artifact, notebook hoặc tài liệu dài, có thể dùng custom agent
`dataset_artifact_reader` hoặc `research_reader`. Sau khi sửa hạ tầng agent, chạy:

```bash
uv run python .agents/scripts/01_validate_workspace.py --full
```

## Các ràng buộc quan trọng

- Đây là pipeline credit scoring thực hành trên GiveMeSomeCredit, không phải hệ thống
  phê duyệt tín dụng production.
- Dataset không có cột thời gian. Split hiện tại là stratified random 60/20/20, không
  được mô tả như out-of-time validation.
- `datasets/`, `outputs/`, `.env` và secret không được commit. Không in
  `KAGGLE_API_TOKEN` ra log.
- Metric và scorecard phải truy vết được tới input, split, config và artifact; không
  suy diễn hiệu quả thực tế chỉ từ một benchmark Kaggle.

## Quyền tự chủ

Tự khảo sát, sửa code/docs, chạy test và dọn artifact trong workspace khi cần cho mục
tiêu đã giao. Luôn bảo toàn thay đổi đang dở. Hỏi trước khi commit, push, xóa dữ liệu
khó phục hồi hoặc thực hiện hành động bên ngoài workspace.

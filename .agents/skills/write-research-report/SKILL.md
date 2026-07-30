---
name: write-research-report
description: Turn code reading, dataset inspection, artifacts, papers, and technical research into reusable Markdown reports or notes. Use when the user says "viết báo cáo", "ghi lại thành md", "tổng hợp thông tin", "đọc hiểu rồi ghi chú", "lưu để đọc lại", "update docs", "so sánh dataset/model/paper", or asks to preserve findings, experiment results, schema analysis, benchmarks, limitations, and sources for later retrieval.
---

# Write research report

Đọc `.agents/workflows/02_write_research_report.md` và `.agents/03_conventions.md` trước.

## Thực hiện

1. Chốt câu hỏi chính và tìm file `docs/` hiện có trước khi tạo file mới.
2. Thu evidence từ runtime/code/artifact và nguồn gốc; không dựa vào tên gọi hoặc roadmap.
3. Với tài liệu dài, dùng `.agents/agents/02_research_reader.md` để nhận evidence map.
4. Viết concept-first; chỉ giữ chi tiết làm thay đổi kết luận hoặc giúp tái lập.
5. Tách rõ verified, inferred, unknown và planned.
6. Kiểm link/path/source và tránh lặp sự thật đã nằm ở file khác.

`docs/` chứa báo cáo đã tổng hợp; `notes/` chứa ghi chú thô; `.agents/memory/` chỉ chứa
preference/correction. Không dùng memory làm kho domain knowledge.

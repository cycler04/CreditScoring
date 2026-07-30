---
name: project-context
---

# Project context

Giai đoạn hiện tại là hoàn thiện pipeline thực hành GiveMeSomeCredit từ dữ liệu raw tới
EDA, baseline model, LR-WoE, scorecard và báo cáo metric.

Ưu tiên dài hạn:

1. tái lập dữ liệu, dependency và lần chạy;
2. ngăn leakage và ghi rõ giới hạn của random split;
3. kiểm chứng metric, WoE/IV, bins, score và cutoff;
4. lưu provenance đủ để audit artifact;
5. mở rộng validation/monitoring trước khi nói tới production.

Trạng thái code hiện tại luôn đọc ở `../01_overview.md` và repository, không lấy từ
memory này.

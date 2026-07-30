# Kế hoạch validation pipeline

## Mục tiêu

Chứng minh pipeline GiveMeSomeCredit có thể tái lập, không leakage và sinh metric cùng
scorecard nhất quán.

## Các bước

1. Xác minh checksum/schema của input và quy tắc clean.
2. Kiểm tra split 60/20/20, stratification, random seed và tính độc lập giữa các tập.
3. Unit test AUC/Gini/KS, PSI, WoE/IV, bin mapping và score transformation.
4. Chạy smoke pipeline trên sample nhỏ trước khi chạy full.
5. Lưu command, environment, input identity, metric và artifact path.
6. Ghi rõ giới hạn do không có time split và dữ liệu population mới.

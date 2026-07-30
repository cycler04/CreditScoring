# Tổng quan workspace

## Mục tiêu

CreditScoring là pipeline có thể tái lập cho bộ dữ liệu GiveMeSomeCredit, bao gồm:

1. tải và kiểm tra dữ liệu;
2. EDA và phân tích missing/anomaly/bad rate;
3. huấn luyện Logistic Regression, LightGBM và Logistic Regression với WoE;
4. đánh giá AUC, Gini, KS và PSI;
5. sinh scorecard, cutoff và artifact báo cáo.

Repo đồng thời lưu tài liệu nền tảng và notebook Kaggle tham khảo. Claim về mô hình
phải phân biệt rõ bằng chứng từ pipeline local với thông tin tham khảo bên ngoài.

## Trạng thái và giới hạn

- Git branch chính là `main`; không tự commit hoặc push.
- Python được khóa ở 3.14; dependency được khóa bằng `uv.lock`.
- Dữ liệu và output không nằm trong Git checkout; cần tải dữ liệu trước khi chạy full.
- GiveMeSomeCredit không có cột thời gian, vì vậy pipeline dùng stratified random split
  60/20/20. PSI trên split này không thay thế monitoring theo thời gian.
- Kết quả Kaggle hoặc metric offline không đủ để khẳng định mô hình phù hợp production,
  công bằng, tuân thủ pháp lý hoặc ổn định theo population mới.

Xem [kiến trúc](02_architecture.md) và [các lệnh đã chuẩn hóa](04_commands.md) trước khi
thay đổi pipeline.

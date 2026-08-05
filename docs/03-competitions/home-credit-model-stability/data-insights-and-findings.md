# HCMS — Data insights và findings

## Kết luận ngắn

**Verified:** train có 1.526.659 case, bad rate 3,1437% qua 92 tuần. Bad rate thay đổi
đáng kể giữa các block: train 3,1254%, valid 4,2548%, OOT test 2,1879%. Vì vậy random
split không thay thế kiểm tra theo tuần. Thêm family lịch sử làm stability LightGBM tăng
từ 0,4682 (A) lên 0,5920 (B) và 0,6322 (C).

| Stage | OOT test AUC | OOT test Gini | Stability |
| --- | ---: | ---: | ---: |
| A | 0,74956 | 0,49911 | 0,46816 |
| B | 0,80745 | 0,61489 | 0,59195 |
| C | 0,83098 | 0,66197 | 0,63225 |

Đây là evidence trực tiếp rằng history family B/C bổ sung tín hiệu trên snapshot này;
không chứng minh causal value hoặc ổn định trên population tương lai.

## Findings

1. **Class imbalance:** accuracy baseline sẽ gây hiểu nhầm; cần AUC/Gini, KS,
   calibration và cutoff outcomes.
2. **Temporal population shift:** bad rate valid/test khác train; metric tổng hợp phải
   đi kèm Gini từng tuần, slope và residual volatility.
3. **Depth-2 chi phối volume:** riêng `credit_bureau_a_2` có 188,3 triệu dòng; bounded
   aggregation là yêu cầu kiến trúc, không chỉ optimization.
4. **Feature breadth có lợi:** Stage B tạo bước nhảy lớn nhất; Stage C tiếp tục cải thiện.
5. **Random split không mặc định dễ hơn:** artifact hiện tại cho random test AUC 0,81828,
   thấp hơn OOT test 0,83098 do khác composition. Điều này không hợp thức hóa random
   validation; nó cho thấy AUC phụ thuộc population.
6. **Recent-only không cải thiện rõ:** training nửa tuần gần nhất đạt stability 0,63222,
   gần như all-train 0,63225; chưa có evidence để bỏ lịch sử train cũ.

## Giới hạn

- Public test local chỉ 10 dòng; không dùng nó để suy ra distribution hidden test.
- Feature importance là model-specific và không phải quan hệ nhân quả.
- Các bảng person/registry có thể chứa sensitive attribute hoặc proxy; pipeline thực
  hành không phải hệ thống phê duyệt production.
- Metric local truy từ `outputs/hcms/run_summary.json`; thay input/config phải sinh lại.


# 6. Metrics, validation, monitoring

## 6.1 Vì sao không dùng accuracy

Bad rate 8% → mô hình đoán "tất cả đều tốt" đạt accuracy 92% và vô dụng hoàn toàn. Notebook "Start Here" nói thẳng ở mục *Metric: ROC AUC*: bài toán này imbalanced nên dùng ROC AUC.

## 6.2 Bộ metric cần dùng

### AUC / Gini
```
Gini = 2 × AUC − 1
```
AUC = xác suất mô hình cho một khách xấu ngẫu nhiên điểm rủi ro cao hơn một khách tốt ngẫu nhiên. Thuần **ranking**, không quan tâm calibration.

Mốc tham chiếu cho application scorecard:

| AUC | Gini | Đánh giá |
|---|---|---|
| 0.50 | 0.00 | Ngẫu nhiên |
| 0.65 | 0.30 | Yếu, dùng tạm |
| 0.70–0.75 | 0.40–0.50 | Bình thường trong sản xuất |
| 0.75–0.80 | 0.50–0.60 | Tốt |
| > 0.85 | > 0.70 | **Nghi leakage** |

### KS (Kolmogorov–Smirnov)
```python
ks = max(tpr - fpr)     # chính là Youden's J
```
Khoảng cách lớn nhất giữa phân phối tích lũy của good và bad. Ngành tín dụng báo cáo KS song song AUC. KS 30–40% là bình thường.

### IV (Information Value)
Đo sức dự báo của **từng biến** (không phải mô hình). Ngưỡng: <0.02 bỏ, 0.02–0.1 yếu, 0.1–0.3 trung bình, 0.3+ mạnh, >0.5 nghi ngờ. Chi tiết ở [05-modeling-playbook.md](05-modeling-playbook.md).

### Calibration
AUC không cho biết PD dự đoán có đúng mức không. Nếu cần PD cho pricing/provisioning: vẽ reliability curve (bad rate thực theo decile PD dự đoán), dùng Platt scaling hoặc isotonic. Không cần nếu chỉ xếp hạng và cắt theo approval rate.

### PSI (Population Stability Index)
```
PSI = Σ (%actual_bin − %expected_bin) × ln(%actual_bin / %expected_bin)
```

| PSI | Ý nghĩa | Hành động |
|---|---|---|
| < 0.10 | Không đổi | Theo dõi bình thường |
| 0.10 – 0.25 | Dịch chuyển nhẹ | Điều tra |
| > 0.25 | Dịch chuyển lớn | Xem xét re-develop |

Áp dụng cho hai thứ:
- **Score PSI** — phân phối điểm đầu ra dịch chuyển.
- **Feature PSI (CSI)** — từng feature dịch chuyển. Đây là thứ cho biết *tại sao* điểm dịch.

PSI đo **thay đổi phân phối**, không đo suy giảm hiệu năng. Phân phối có thể đổi mà mô hình vẫn tốt (khách tốt hơn), hoặc phân phối giữ nguyên mà quan hệ feature–target đổi. Cần đo cả hai.

## 6.3 Stability metric của Home Credit (đáng học)

Đây là công thức lượng hóa "mô hình ổn định" thành một con số. Chép nguyên từ trang Evaluation:

1. Tính gini cho từng `WEEK_NUM`.
2. Fit hồi quy tuyến tính `a·x + b` qua chuỗi gini theo tuần.
3. `falling_rate = min(0, a)` — chỉ phạt khi dốc xuống.
4. Lấy std của phần dư (residuals) quanh đường hồi quy.

```
stability metric = mean(gini) + 88.0 × min(0, a) − 0.5 × std(residuals)
```

Đọc công thức:
- Hệ số **88** cực lớn → xu hướng giảm bị phạt rất nặng. Gini giảm 0.001/tuần đã trừ 0.088 khỏi điểm, tương đương mất 0.088 mean gini (≈ 0.044 AUC). Đây là tuyên bố rõ ràng: thà gini thấp mà phẳng còn hơn cao mà tụt.
- Hệ số **0.5** cho std → dao động bị phạt nhẹ hơn xu hướng giảm. Nhiễu tuần này qua tuần khác chấp nhận được; suy giảm có hệ thống thì không.

Có thể áp dụng lại cho dự án nội bộ mà không cần dữ liệu Home Credit: chỉ cần một cột thời gian và đủ mẫu mỗi kỳ (~vài trăm hồ sơ/tuần, trong đó có bad).

## 6.4 Bảng monitoring tối thiểu khi lên production

| Chỉ số | Tần suất | Ngưỡng cảnh báo |
|---|---|---|
| Score PSI vs tập dev | Tháng | > 0.10 điều tra, > 0.25 báo động |
| Feature PSI (top 20 feature) | Tháng | như trên |
| Gini theo tháng (khi label đã chín) | Tháng | giảm > 10% so với dev |
| Bad rate thực vs dự đoán theo grade | Tháng | lệch > 20% |
| Approval rate | Tuần | lệch bất thường |
| Tỷ lệ missing / lỗi từng nguồn dữ liệu | Ngày | tăng đột biến |

Dòng cuối quan trọng nhất mà hay bị bỏ: mô hình thường hỏng do **pipeline dữ liệu** (nguồn ngoài chết, đổi format, đổi mã category) chứ không do khách hàng đổi hành vi. Home Credit nói trước điều này trong mô tả dữ liệu: nguồn external có thể không còn khả dụng ở kỳ đánh giá tương lai.

## 6.5 Độ trễ label — hệ quả thực tế

Với performance window 12 tháng, gini thật của hồ sơ tháng này chỉ biết được sau ~12 tháng. Nghĩa là:
- Monitoring theo hiệu năng luôn **chậm ít nhất một performance window**.
- Nên dùng chỉ báo sớm: PSI (có ngay), bad rate ở 3 tháng đầu (early vintage), tỷ lệ quá hạn kỳ đầu (FPD — first payment default).
- FPD là chỉ báo nhanh nhất: khách trượt ngay kỳ trả đầu tiên. Có sau 30–60 ngày.

# 2. Label (target) đến từ đâu

Điểm quan trọng nhất của tuần này: **label không có trong dữ liệu, phải tự tạo ra.** Nó là kết quả của một loạt quyết định nghiệp vụ, và làm sai chỗ này thì mọi thứ phía sau vô nghĩa.

## 2.1 Nguồn gốc vật lý của label

Label sinh ra từ **hệ thống core banking / loan management**, bảng lịch trả nợ (repayment schedule) đối chiếu với giao dịch thực tế:

```
loan_id | due_date   | due_amount | paid_date  | paid_amount
L001    | 2025-01-10 | 1,000,000  | 2025-01-09 | 1,000,000   → DPD 0
L001    | 2025-02-10 | 1,000,000  | 2025-03-25 | 1,000,000   → DPD 43
L001    | 2025-03-10 | 1,000,000  | NULL       | 0            → DPD đang chạy
```

Từ đó tính **DPD (Days Past Due)** cho mỗi kỳ, rồi max DPD trong một cửa sổ thời gian → so với ngưỡng → ra label 0/1.

## 2.2 Định nghĩa "default" chuẩn ngành

Basel: khách hàng default khi xảy ra một trong hai:
1. **90+ ngày quá hạn** với nghĩa vụ tín dụng trọng yếu, hoặc
2. **Unlikeliness to pay** — ngân hàng đánh giá khách không có khả năng trả đủ (đã bán nợ, xóa nợ, tái cơ cấu do khó khăn, phá sản...).

Kaggle "Give Me Some Credit" dùng đúng ngưỡng này — biến target là `SeriousDlqin2yrs`, định nghĩa: *"Person experienced 90 days past due delinquency or worse"*.

Các ngưỡng khác dùng trong thực tế:
- **30+ DPD** — early risk indicator, nhiều bad hơn, model học nhanh nhưng lẫn nhiều người chỉ quên trả.
- **60+ DPD** — trung gian.
- **90+ DPD** — chuẩn Basel, mặc định nên chọn.
- **Charge-off / write-off** — chắc chắn xấu nhưng phát hiện rất muộn (12–24 tháng).

## 2.3 Observation window vs Performance window

```
        observation window            performance window
 |<------------------------->|<--------------------------------->|
 ...lịch sử khách hàng...   T0                                  T0+12m
                        (điểm quyết định)                    (chốt label)
     → FEATURE lấy ở đây     |            → LABEL lấy ở đây
```

Quy tắc bất di bất dịch:
- **Mọi feature phải tính bằng dữ liệu có trước T0.** Không được dùng bất cứ thứ gì phát sinh sau T0.
- **Label = có default xảy ra trong (T0, T0+W] hay không**, W thường 12 tháng (đôi khi 18–24 với sản phẩm dài hạn).

Chọn W bằng **vintage analysis**: vẽ tỷ lệ default tích lũy theo số tháng kể từ giải ngân (cohort theo tháng giải ngân). Đường cong thường dốc đến tháng 9–12 rồi phẳng dần → chọn W tại điểm bắt đầu phẳng. Ngắn quá thì bỏ sót bad; dài quá thì mất dữ liệu gần đây (chưa đủ chín) và giảm số mẫu.

Home Credit Model Stability mô tả label đúng theo logic này: *"target — determined after a certain period based on whether or not the client defaulted"*, và `date_decision` chính là T0.

## 2.4 Các nhóm khách bị loại/gán riêng (exclusions & indeterminate)

Không phải dòng nào cũng vào tập train:

| Nhóm | Xử lý |
|---|---|
| Chưa đủ performance window (khoản vay quá mới) | Loại khỏi train (immature) |
| Tất toán trước hạn rất sớm | Thường gán good, cần kiểm tra |
| Rơi vào vùng "indeterminate" (ví dụ 30–89 DPD) | Loại, hoặc gán good — phải nhất quán và ghi lại |
| Gian lận (fraud) | Loại — fraud là mô hình khác |
| Hồ sơ bị từ chối | Không có label → xem 2.6 |

## 2.5 Hai cách map label trong thực tế — ví dụ đối chiếu

**Give Me Some Credit** (đơn giản, đã đóng gói sẵn):
`SeriousDlqin2yrs` ∈ {0,1}, 1 = có 90+ DPD trong 2 năm.

**LendingClub** (notebook Beata Faron tự map từ `loan_status`):
```python
loan_status_mapping = {
    'Fully Paid': 1, 'Current': 1, 'In Grace Period': 1,
    'Late (16-30 days)': 0, 'Late (31-120 days)': 0,
    'Charged Off': 0, 'Default': 0
}
```
Lưu ý ba điểm ở ví dụ này:
1. Quy ước **ngược** với thông lệ: 1 = good, 0 = bad. Phải cực kỳ cẩn thận khi đọc WoE/IV, vì công thức WoE đổi dấu theo quy ước.
2. `Current` (đang trả bình thường) được gán good — nhưng khoản vay đó chưa chạy hết đời, có thể xấu sau. Đây là **snapshot label**, không phải label theo performance window đúng chuẩn.
3. Ngưỡng bad ở đây là **16+ DPD**, không phải 90+ → bad rate cao hơn chuẩn Basel.

Bad rate hai bộ dữ liệu để tham chiếu:
- Home Credit Default Risk: TARGET=1 chiếm ~8% (imbalanced rõ).
- LendingClub trong notebook: 325,071 / 2,029,952 ≈ 16% ở lớp thiểu số.

## 2.6 Reject inference (biết để đó, chưa làm tuần này)

Bạn chỉ có label cho những hồ sơ **đã được duyệt**. Những hồ sơ bị từ chối không có kết quả → tập train bị chệch (selection bias), mô hình chỉ học được trên vùng khách đã qua sàng lọc trước đó. Kỹ thuật xử lý: parcelling, augmentation, fuzzy augmentation, hoặc dùng bureau outcome của khách bị từ chối. Toàn bộ 7 link Kaggle đều **không** đụng tới vấn đề này — đó là khoảng cách lớn giữa Kaggle và production.

## 2.7 Checklist khi nhận dữ liệu thật

- [ ] Bảng nào chứa lịch trả nợ? Ai là owner?
- [ ] DPD được tính sẵn hay phải tự tính từ due_date/paid_date?
- [ ] Chọn ngưỡng default nào? (mặc định: 90+ DPD)
- [ ] Performance window bao lâu? Có chạy được vintage analysis không?
- [ ] Nhóm nào bị loại khỏi train? Ghi lại thành rule.
- [ ] Bad rate là bao nhiêu? Có đủ số bad để train không? (rule of thumb: ≥ 1,000–2,000 bad)
- [ ] Bad rate có ổn định theo tháng giải ngân không? Nếu nhảy đột ngột → có thay đổi chính sách duyệt.
- [ ] Quy ước 1 = bad hay 1 = good? Viết vào data dictionary.

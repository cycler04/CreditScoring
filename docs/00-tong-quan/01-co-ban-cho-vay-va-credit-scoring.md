# 1. Cơ bản về cho vay và credit scoring

## 1.1 Bài toán kinh doanh

Người cho vay đối mặt hai loại lỗi, chi phí **không đối xứng**:

| Quyết định | Thực tế tốt | Thực tế xấu |
|---|---|---|
| Duyệt | Lãi thu được (~vài % dư nợ) | Mất gốc (~40–80% dư nợ) |
| Từ chối | Mất cơ hội (opportunity cost) | Tránh được lỗ |

Hệ quả: một khoản bad "ăn" lợi nhuận của ~10–30 khoản good. Vì vậy mô hình phải xếp hạng rủi ro (ranking) tốt, và cutoff phải chọn theo tiền, không theo accuracy.

Notebook LendingClub (report #3) phát biểu đúng ý này: duyệt nhầm khách xấu = lỗ tài chính; từ chối nhầm khách tốt = mất doanh thu.

## 1.2 Ba tham số rủi ro tín dụng

Expected Loss:

```
EL = PD × LGD × EAD
```

- **PD** (Probability of Default) — xác suất vỡ nợ trong 12 tháng tới. Đây là thứ credit scoring model dự đoán.
- **LGD** (Loss Given Default) — tỷ lệ mất vốn khi đã vỡ nợ (sau thu hồi, xử lý tài sản bảo đảm).
- **EAD** (Exposure At Default) — dư nợ tại thời điểm vỡ nợ.

Pre-Sprint 0 và toàn bộ 7 link Kaggle trong task.txt đều **chỉ nói về PD**. LGD/EAD là bài toán riêng.

## 1.3 Các loại scorecard theo thời điểm dùng

| Loại | Dùng khi | Dữ liệu có sẵn |
|---|---|---|
| **Application scorecard** | Lúc khách nộp hồ sơ | Thông tin khai báo + bureau + hồ sơ cũ |
| **Behavioral scorecard** | Khách đang vay, chấm lại định kỳ | Thêm lịch sử trả nợ của chính khoản vay đó |
| **Collection scorecard** | Khách đã quá hạn | Thêm hành vi thu hồi nợ |

Phân biệt này quan trọng vì nó quyết định feature nào được phép dùng:
- Home Credit Default Risk / GiveMeSomeCredit / Model Stability = **application** scorecard.
- Notebook LendingClub của Beata Faron = **behavioral** scorecard (dùng `total_rec_prncp`, `last_pymnt_amnt`, `recoveries`... — chỉ tồn tại sau giải ngân). Xem cảnh báo leakage ở [04-eda-playbook.md](04-eda-playbook.md).

## 1.4 Scorecard là gì (dạng sản phẩm cuối)

Không giao "xác suất 0.037" cho bộ phận tín dụng. Giao một bảng điểm:

```
Feature            Bin                 Điểm
age                18–25                 +12
age                26–35                 +28
age                36+                   +41
credit_history     < 1 năm               +5
credit_history     1–5 năm               +22
...
                                   Tổng: 300–900
```

Công thức chuẩn ngành (notebook WoE & Scorecard dùng đúng dạng này):

```
Score = Base + Factor × ln(odds)
      = Base + Factor × ln((1 − PD) / PD)
```

- `Factor = PDO / ln(2)` với PDO = Points to Double the Odds (thường 20).
- Điểm cao = rủi ro thấp.
- Vì logistic regression cộng tuyến tính theo log-odds, mỗi bin trở thành một số điểm cố định → giải thích được với khách hàng và với thanh tra.

## 1.5 Bối cảnh pháp lý (Basel II)

Notebook LendingClub tóm gọn: Basel II yêu cầu ngân hàng (i) giữ vốn tương ứng rủi ro, (ii) chịu supervisory review, (iii) minh bạch thị trường. Có 3 mức tiếp cận: Standardised Approach (SA), Foundation IRB, Advanced IRB. Ở IRB, ngân hàng tự ước lượng PD (và LGD/EAD ở A-IRB) → mô hình phải được validate và giám sát định kỳ.

Hệ quả thực tế cho ML:
- Mô hình phải **giải thích được** → đây là lý do WoE + Logistic Regression vẫn sống khỏe dù AUC thấp hơn GBDT.
- Phải có quy trình **monitoring** (PSI, gini theo thời gian) và **re-development** định kỳ.
- Không được dùng biến phân biệt đối xử (giới tính, chủng tộc, tôn giáo...) — Home Credit Default Risk có `CODE_GENDER`, dùng được trên Kaggle nhưng không dùng được trong production ở nhiều thị trường.

## 1.6 Vì sao stability quan trọng

Trích thẳng mô tả cuộc thi Home Credit - Credit Risk Model Stability:

> "clients' behaviors change constantly, so every scorecard must be updated regularly... The core of the issue is that loan providers aren't able to spot potential problems any sooner than the first due dates of those loans are observable."

Nghĩa là: bạn chỉ biết mô hình hỏng sau khi đã cho vay xấu vài tháng. Giữa thời điểm phát hiện và thời điểm thay mô hình mới còn cả quá trình redevelop + validate + deploy. Vì vậy có trade-off **performance ↔ stability**, và đôi khi phải chọn mô hình AUC thấp hơn nhưng ít suy giảm hơn.

## Đọc tiếp

- Label từ đâu → [02-label-target-den-tu-dau.md](02-label-target-den-tu-dau.md)
- Feature từ đâu → [03-feature-den-tu-dau.md](03-feature-den-tu-dau.md)

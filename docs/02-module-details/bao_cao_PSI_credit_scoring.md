# Báo cáo chi tiết về PSI trong Credit Scoring

## 1. Tổng quan

**PSI (Population Stability Index)** là chỉ số dùng để đo mức độ thay đổi của phân phối dữ liệu giữa hai tập dữ liệu:

- **Tập tham chiếu (Expected / Reference population)**: thường là dữ liệu phát triển mô hình, dữ liệu huấn luyện, hoặc dữ liệu của một giai đoạn chuẩn.
- **Tập hiện tại (Actual / Current population)**: dữ liệu phát sinh sau khi mô hình được triển khai, ví dụ dữ liệu theo tháng hoặc quý.

Trong credit scoring, PSI được dùng chủ yếu để giám sát:

1. Phân phối điểm tín dụng.
2. Phân phối xác suất vỡ nợ.
3. Phân phối của từng biến đầu vào.
4. Sự thay đổi của tập khách hàng được phê duyệt.
5. Sự thay đổi giữa các kênh, sản phẩm hoặc nhóm khách hàng.

PSI không đo trực tiếp độ chính xác của mô hình. Nó chỉ cho biết dữ liệu hoặc điểm số đã thay đổi bao nhiêu so với giai đoạn tham chiếu.

---

## 2. Vì sao PSI quan trọng trong credit scoring?

Mô hình credit scoring được xây dựng dựa trên giả định rằng dữ liệu trong tương lai không khác quá nhiều so với dữ liệu lịch sử.

Tuy nhiên, phân phối khách hàng có thể thay đổi do:

- Điều kiện kinh tế thay đổi.
- Chính sách phê duyệt thay đổi.
- Chiến dịch marketing mới.
- Ra mắt sản phẩm mới.
- Thay đổi trong nguồn dữ liệu.
- Thay đổi quy trình thu thập thông tin.
- Thay đổi định nghĩa biến.
- Thay đổi hành vi khách hàng.
- Gian lận hoặc thao túng hồ sơ.
- Tác động của mùa vụ.
- Sự thay đổi trong nhóm khách hàng mục tiêu.

Ví dụ:

- Thu nhập trung bình của khách hàng mới thấp hơn trước.
- Tỷ lệ khách hàng vay tín chấp tăng mạnh.
- Khoản vay trung bình tăng.
- Điểm tín dụng dịch chuyển về phía rủi ro cao.
- Một biến bị lỗi ETL khiến nhiều giá trị bị gán bằng 0.

PSI giúp phát hiện sớm những thay đổi này trước khi hiệu quả mô hình suy giảm nghiêm trọng.

---

## 3. Công thức PSI

Giả sử một biến được chia thành \(n\) nhóm hoặc bin.

Công thức PSI:

\[
PSI = \sum_{i=1}^{n}(A_i - E_i)\ln\left(\frac{A_i}{E_i}\right)
\]

Trong đó:

- \(E_i\): tỷ lệ quan sát trong bin \(i\) của tập tham chiếu.
- \(A_i\): tỷ lệ quan sát trong bin \(i\) của tập hiện tại.
- \(\ln\): logarit tự nhiên.

PSI của toàn bộ biến bằng tổng PSI của tất cả các bin.

### PSI theo từng bin

\[
PSI_i = (A_i - E_i)\ln\left(\frac{A_i}{E_i}\right)
\]

Mỗi bin đóng góp một phần vào PSI tổng.

Nếu phân phối hiện tại giống hoàn toàn phân phối tham chiếu:

\[
A_i = E_i
\]

thì:

\[
PSI_i = 0
\]

và PSI tổng bằng 0.

---

## 4. Cách hiểu trực quan

PSI đo hai thành phần:

1. **Độ chênh lệch tuyệt đối** giữa tỷ lệ hiện tại và tỷ lệ tham chiếu.
2. **Độ chênh lệch tương đối** thông qua logarit của tỷ lệ \(A_i/E_i\).

Vì vậy, một bin có tỷ lệ thay đổi mạnh sẽ đóng góp lớn vào PSI.

Ví dụ:

| Bin | Expected | Actual | Nhận xét |
|---|---:|---:|---|
| 1 | 10% | 10% | Không thay đổi |
| 2 | 10% | 11% | Thay đổi nhỏ |
| 3 | 10% | 20% | Thay đổi lớn |
| 4 | 10% | 2% | Thay đổi rất lớn |

Các bin 3 và 4 sẽ đóng góp PSI lớn hơn nhiều so với bin 2.

---

## 5. Ngưỡng diễn giải PSI

Một quy tắc thực hành phổ biến:

| PSI | Diễn giải |
|---:|---|
| < 0.10 | Phân phối tương đối ổn định |
| 0.10 – 0.25 | Có sự thay đổi đáng kể, cần theo dõi |
| > 0.25 | Thay đổi lớn, cần điều tra |
| > 0.50 | Drift rất mạnh, có thể có lỗi dữ liệu hoặc thay đổi cấu trúc |

Các ngưỡng này chỉ là **rule of thumb**. Không nên áp dụng máy móc cho mọi mô hình.

Ngưỡng phù hợp phụ thuộc vào:

- Loại sản phẩm.
- Tần suất giám sát.
- Quy mô dữ liệu.
- Mức biến động tự nhiên.
- Chính sách quản trị mô hình.
- Yêu cầu của tổ chức hoặc cơ quan quản lý.
- Tầm quan trọng của biến.
- Mức độ rủi ro của quyết định tín dụng.

Một biến có PSI bằng 0.12 chưa chắc là vấn đề nếu biến đó thường có tính mùa vụ. Ngược lại, PSI bằng 0.08 có thể đáng chú ý nếu biến đó vốn rất ổn định.

---

## 6. Các loại PSI trong credit scoring

## 6.1. Score PSI

Score PSI đo sự thay đổi trong phân phối điểm tín dụng.

Ví dụ:

- Điểm scorecard.
- Điểm từ 300 đến 850.
- Điểm nội bộ từ 0 đến 1000.
- Điểm log-odds.
- Xác suất vỡ nợ sau khi chuyển thành score.

Score PSI thường là chỉ số PSI quan trọng nhất trong dashboard giám sát mô hình.

Score PSI cao có thể phản ánh:

- Tập khách hàng hiện tại rủi ro hơn.
- Chính sách cấp tín dụng thay đổi.
- Đầu vào của mô hình bị drift.
- Mô hình đang tạo ra phân phối điểm khác trước.
- Kênh hoặc sản phẩm mới làm thay đổi hồ sơ khách hàng.

---

## 6.2. Variable PSI

Variable PSI đo sự thay đổi phân phối của từng biến đầu vào.

Ví dụ:

- Tuổi.
- Thu nhập.
- Dư nợ.
- Tỷ lệ sử dụng hạn mức.
- Số lần trễ hạn.
- Số tài khoản tín dụng.
- Thời gian làm việc.
- Số lần truy vấn CIC.
- Tỷ lệ nợ trên thu nhập.
- Loại hình nghề nghiệp.

Variable PSI giúp xác định nguyên nhân của score drift.

Ví dụ:

| Biến | PSI |
|---|---:|
| Age | 0.03 |
| Income | 0.06 |
| Loan Amount | 0.31 |
| Debt-to-Income | 0.14 |
| Delinquency Count | 0.05 |

Trong ví dụ trên, `Loan Amount` là biến cần điều tra đầu tiên.

---

## 6.3. PSI cho probability of default

Nếu mô hình xuất ra xác suất vỡ nợ, có thể tính PSI trực tiếp trên:

- Raw probability.
- Calibrated probability.
- Logit score.
- Risk band.

Cần cố định bin theo tập tham chiếu, không tạo lại bin riêng cho từng tháng.

---

## 6.4. PSI theo segment

Có thể tính PSI riêng theo:

- Sản phẩm.
- Kênh bán.
- Khu vực.
- Chi nhánh.
- Nhóm thu nhập.
- Loại khách hàng.
- Khách hàng mới và khách hàng hiện hữu.
- Secured và unsecured lending.
- Pre-approved và non-pre-approved.
- Approved và declined population.

PSI tổng có thể thấp nhưng một segment cụ thể có PSI rất cao.

---

## 6.5. PSI theo thời gian

PSI thường được tính:

- Hàng tuần.
- Hàng tháng.
- Hàng quý.
- Rolling 30 ngày.
- Rolling 90 ngày.
- So với dữ liệu phát triển.
- So với tháng trước.
- So với trung bình 12 tháng.

Các cách so sánh khác nhau phục vụ mục tiêu khác nhau.

### So với dữ liệu phát triển

Phù hợp để kiểm tra mô hình đã khác bao nhiêu so với thời điểm xây dựng.

### So với tháng trước

Phù hợp để phát hiện thay đổi đột ngột.

### So với rolling baseline

Phù hợp cho môi trường có tính mùa vụ hoặc biến động liên tục.

---

## 7. Quy trình tính PSI

## Bước 1: Chọn tập tham chiếu

Tập tham chiếu nên đại diện cho trạng thái bình thường của mô hình.

Có thể sử dụng:

- Development sample.
- Training sample.
- Validation sample.
- Dữ liệu tại thời điểm triển khai.
- Trung bình của nhiều tháng ổn định.
- Dữ liệu của cùng kỳ năm trước.

Không nên chọn tập tham chiếu quá nhỏ hoặc không đại diện.

---

## Bước 2: Xác định bin

Các cách chia bin phổ biến:

### Quantile binning

Chia biến thành các nhóm có số lượng gần bằng nhau trên tập tham chiếu.

Ví dụ:

- 10 bin tương ứng với decile.
- 20 bin tương ứng với ventile.

Ưu điểm:

- Mỗi bin có đủ quan sát.
- Dễ so sánh.

Nhược điểm:

- Biên bin có thể khó giải thích.
- Không phù hợp nếu biến có nhiều giá trị trùng nhau.

### Equal-width binning

Chia khoảng giá trị thành các đoạn có độ rộng bằng nhau.

Ưu điểm:

- Dễ hiểu.
- Dễ trình bày.

Nhược điểm:

- Một số bin có thể chứa rất ít quan sát.
- Nhạy với outlier.

### Business binning

Chia bin theo ngưỡng nghiệp vụ.

Ví dụ:

- Tuổi: dưới 25, 25–34, 35–44, 45–54, trên 55.
- DTI: dưới 20%, 20–40%, 40–60%, trên 60%.
- Delinquency: 0, 1, 2, từ 3 trở lên.

Ưu điểm:

- Dễ giải thích.
- Phù hợp scorecard truyền thống.

### WOE binning

Sử dụng chính các bin đã được dùng trong scorecard.

Đây thường là lựa chọn phù hợp nhất với logistic scorecard vì:

- Phản ánh đúng cấu trúc mô hình.
- Dễ liên kết với WOE và coefficient.
- Dễ truy nguyên tác động đến score.

---

## Bước 3: Áp dụng cùng biên bin

Biên bin phải được xác định trên tập tham chiếu và giữ cố định.

Không được chia lại bin riêng trên tập hiện tại, vì khi đó các bin luôn chứa tỷ lệ tương tự nhau và PSI sẽ bị sai lệch.

Ví dụ đúng:

1. Tạo decile từ dữ liệu training.
2. Lưu biên của 10 bin.
3. Áp dụng đúng các biên đó cho dữ liệu tháng 1, tháng 2, tháng 3.

---

## Bước 4: Tính tỷ lệ từng bin

Với mỗi bin:

\[
E_i = \frac{\text{Số quan sát tham chiếu trong bin } i}{\text{Tổng số quan sát tham chiếu}}
\]

\[
A_i = \frac{\text{Số quan sát hiện tại trong bin } i}{\text{Tổng số quan sát hiện tại}}
\]

---

## Bước 5: Xử lý tỷ lệ bằng 0

Nếu \(E_i = 0\) hoặc \(A_i = 0\), công thức logarit không xác định.

Cách phổ biến là thay tỷ lệ 0 bằng một số nhỏ:

\[
\epsilon = 0.0001
\]

hoặc:

\[
\epsilon = 10^{-6}
\]

Ví dụ:

```text
adjusted_expected = max(expected_pct, 0.0001)
adjusted_actual = max(actual_pct, 0.0001)
```

Giá trị epsilon cần được chuẩn hóa trong toàn bộ hệ thống giám sát.

---

## Bước 6: Tính PSI theo bin

\[
PSI_i = (A_i - E_i)\ln\left(\frac{A_i}{E_i}\right)
\]

---

## Bước 7: Cộng PSI

\[
PSI_{total} = \sum_i PSI_i
\]

---

## 8. Ví dụ tính PSI

Giả sử điểm tín dụng được chia thành 5 bin.

| Score bin | Expected count | Expected % | Actual count | Actual % |
|---|---:|---:|---:|---:|
| 0–200 | 2,000 | 20% | 1,500 | 15% |
| 200–400 | 2,000 | 20% | 1,800 | 18% |
| 400–600 | 2,000 | 20% | 2,500 | 25% |
| 600–800 | 2,000 | 20% | 2,700 | 27% |
| 800–1000 | 2,000 | 20% | 1,500 | 15% |

Tính từng bin:

\[
PSI_1 = (0.15 - 0.20)\ln(0.15/0.20)
\]

\[
PSI_1 \approx 0.0144
\]

Tương tự cho các bin còn lại:

| Bin | Expected % | Actual % | PSI contribution |
|---|---:|---:|---:|
| 0–200 | 0.20 | 0.15 | 0.0144 |
| 200–400 | 0.20 | 0.18 | 0.0021 |
| 400–600 | 0.20 | 0.25 | 0.0112 |
| 600–800 | 0.20 | 0.27 | 0.0210 |
| 800–1000 | 0.20 | 0.15 | 0.0144 |

Tổng PSI xấp xỉ:

\[
PSI \approx 0.063
\]

Theo ngưỡng thông thường, phân phối vẫn tương đối ổn định.

---

## 9. Ví dụ PSI cao

Giả sử phân phối điểm thay đổi mạnh:

| Score bin | Expected % | Actual % |
|---|---:|---:|
| Very high risk | 10% | 28% |
| High risk | 20% | 30% |
| Medium risk | 30% | 25% |
| Low risk | 25% | 12% |
| Very low risk | 15% | 5% |

Trong trường hợp này, PSI có thể vượt 0.25.

Điều này cho thấy khách hàng hiện tại đang dịch chuyển rõ rệt về phía rủi ro cao.

Tuy nhiên, PSI chưa cho biết nguyên nhân. Cần kiểm tra thêm:

- Thay đổi kênh acquisition.
- Thay đổi chính sách approval.
- Thay đổi sản phẩm.
- Lỗi dữ liệu.
- Thay đổi macroeconomic.
- Drift của các biến đầu vào.
- Thay đổi tỷ lệ missing.
- Thay đổi tỷ lệ decline.
- Thay đổi mix khách hàng.

---

## 10. PSI không phải là performance metric

PSI không đo:

- ROC-AUC.
- Gini.
- KS.
- Accuracy.
- Precision.
- Recall.
- Calibration.
- Bad rate prediction error.
- Default prediction quality.

Có thể xảy ra các trường hợp sau:

### PSI thấp, performance thấp

Phân phối dữ liệu không thay đổi nhiều nhưng mô hình vốn đã yếu hoặc bị giảm khả năng phân biệt.

### PSI cao, performance vẫn tốt

Khách hàng thay đổi nhưng mô hình vẫn phân loại tốt.

### PSI thấp, calibration xấu

Phân phối score ổn định nhưng xác suất dự báo không còn phản ánh đúng bad rate thực tế.

### PSI cao do thay đổi tích cực

Ví dụ tổ chức tập trung vào khách hàng chất lượng cao, khiến score distribution tốt hơn. PSI vẫn cao nhưng không nhất thiết là vấn đề.

Do đó, PSI phải được giám sát cùng các chỉ số khác.

---

## 11. PSI so với các metric khác

| Metric | Mục tiêu |
|---|---|
| PSI | Đo thay đổi phân phối |
| CSI | Đo thay đổi của characteristic hoặc biến đầu vào |
| KS | Đo khả năng tách good và bad |
| ROC-AUC | Đo khả năng xếp hạng rủi ro |
| Gini | Chuyển đổi từ AUC để đo discrimination |
| IV | Đo sức mạnh dự báo của biến |
| WOE | Biểu diễn quan hệ giữa bin và bad rate |
| Brier Score | Đo sai số xác suất |
| Calibration slope/intercept | Đo độ hiệu chỉnh |
| Bad rate | Đo tỷ lệ khách hàng xấu thực tế |
| Approval rate | Đo tỷ lệ được phê duyệt |
| Override rate | Đo tỷ lệ quyết định bị điều chỉnh thủ công |

### PSI và CSI

Trong một số tổ chức:

- **PSI** dùng cho phân phối điểm hoặc phân phối tổng thể.
- **CSI (Characteristic Stability Index)** dùng cho từng biến đầu vào.

Về mặt công thức, CSI thường giống PSI. Khác biệt chủ yếu nằm ở cách gọi và đối tượng áp dụng.

---

## 12. Hạn chế của PSI

## 12.1. Phụ thuộc vào cách chia bin

PSI có thể thay đổi đáng kể nếu:

- Số lượng bin khác nhau.
- Biên bin khác nhau.
- Sử dụng quantile thay vì equal-width.
- Xử lý missing khác nhau.
- Gộp category khác nhau.

Vì vậy, PSI chỉ có ý nghĩa khi cách binning được cố định.

---

## 12.2. Không có kiểm định thống kê rõ ràng

PSI không trực tiếp cho biết sự thay đổi có ý nghĩa thống kê hay không.

Với mẫu rất lớn, thay đổi nhỏ vẫn có thể đáng tin cậy. Với mẫu nhỏ, PSI có thể dao động mạnh.

Có thể kết hợp với:

- Chi-square test.
- Kolmogorov–Smirnov two-sample test.
- Jensen-Shannon divergence.
- Wasserstein distance.
- Confidence interval.
- Bootstrap.

---

## 12.3. Nhạy với bin có tỷ lệ rất nhỏ

Một bin có tỷ lệ gần 0 có thể tạo contribution lớn.

Cần kiểm tra:

- Outlier.
- Missing.
- Rare category.
- Epsilon.
- Quy tắc gộp bin nhỏ.

---

## 12.4. Không xác định nguyên nhân

PSI chỉ báo drift, không giải thích drift.

Cần drill-down theo:

- Variable.
- Segment.
- Channel.
- Product.
- Region.
- Time.
- Missing rate.
- Category.
- Data source.
- Decision stage.

---

## 12.5. Không phản ánh trực tiếp tác động kinh doanh

PSI cao không đồng nghĩa với:

- Tăng nợ xấu.
- Tăng tổn thất.
- Giảm lợi nhuận.
- Mô hình cần retrain ngay.

Phải kiểm tra thêm các chỉ số nghiệp vụ và performance.

---

## 13. Cách sử dụng PSI đúng trong hệ thống giám sát

Một framework giám sát nên gồm ít nhất:

### Data quality

- Số lượng bản ghi.
- Tỷ lệ missing.
- Tỷ lệ outlier.
- Tỷ lệ invalid.
- Duplicate.
- Mapping lỗi.
- Thay đổi schema.
- Min, max, mean, median.
- Category mới.

### Stability

- Score PSI.
- Variable PSI.
- PSI theo segment.
- PSI theo thời gian.
- Tỷ lệ record ngoài range.
- Tỷ lệ unseen category.

### Performance

- AUC.
- Gini.
- KS.
- Bad rate theo score band.
- Calibration.
- Lift.
- Capture rate.
- Brier Score.

### Business

- Approval rate.
- Decline rate.
- Override rate.
- Booking rate.
- Default rate.
- Loss rate.
- Expected loss.
- Vintage performance.
- Roll rate.

---

## 14. Quy trình điều tra khi PSI tăng

Khi PSI vượt ngưỡng, không nên retrain ngay. Nên điều tra theo trình tự.

### Bước 1: Kiểm tra dữ liệu

- Pipeline có lỗi không?
- Schema có thay đổi không?
- Có nhiều null hơn không?
- Category mới có xuất hiện không?
- Có biến bị scale sai không?
- Có thay đổi đơn vị không?
- Có lỗi mapping không?
- Có thay đổi source system không?

### Bước 2: Kiểm tra volume

- Số lượng hồ sơ có giảm mạnh không?
- Một số bin có quá ít quan sát không?
- PSI cao có phải do sample size nhỏ không?

### Bước 3: Kiểm tra policy

- Cutoff có thay đổi không?
- Chính sách approval có thay đổi không?
- Quy tắc pre-screening có thay đổi không?
- Có chiến dịch mới không?
- Có sản phẩm mới không?

### Bước 4: Kiểm tra segment

- Drift đến từ kênh nào?
- Drift đến từ khu vực nào?
- Drift đến từ sản phẩm nào?
- Drift đến từ customer type nào?

### Bước 5: Kiểm tra performance

- AUC có giảm không?
- KS có giảm không?
- Calibration có xấu đi không?
- Bad rate có tăng không?
- Lift có giảm không?

### Bước 6: Quyết định hành động

Có thể chọn:

- Tiếp tục theo dõi.
- Điều chỉnh threshold.
- Recalibrate probability.
- Sửa data pipeline.
- Thay đổi policy.
- Loại bỏ hoặc thay thế biến.
- Retrain mô hình.
- Redevelop mô hình.
- Xây mô hình riêng cho segment.

---

## 15. Gợi ý ngưỡng cảnh báo

Một hệ thống cảnh báo có thể dùng:

| Mức | Score PSI | Variable PSI | Hành động |
|---|---:|---:|---|
| Green | < 0.10 | < 0.10 | Tiếp tục theo dõi |
| Amber | 0.10–0.25 | 0.10–0.25 | Phân tích nguyên nhân |
| Red | > 0.25 | > 0.25 | Điều tra ngay |
| Critical | > 0.50 | > 0.50 | Kiểm tra lỗi dữ liệu hoặc thay đổi cấu trúc |

Nên bổ sung quy tắc:

- Cảnh báo nếu PSI tăng liên tục 3 kỳ.
- Cảnh báo nếu nhiều biến cùng vượt ngưỡng.
- Cảnh báo nếu score PSI cao cùng lúc với AUC giảm.
- Cảnh báo nếu PSI cao tập trung ở một segment.
- Cảnh báo nếu missing rate tăng bất thường.
- Cảnh báo nếu xuất hiện category mới.

---

## 16. PSI cho biến liên tục

Với biến liên tục:

1. Tạo bin trên tập tham chiếu.
2. Lưu biên bin.
3. Tạo bin riêng cho missing nếu cần.
4. Áp dụng cùng bin lên tập hiện tại.
5. Tính tỷ lệ.
6. Tính PSI.

Ví dụ:

```text
Income bins:
Missing
<= 5 triệu
5–10 triệu
10–20 triệu
20–40 triệu
> 40 triệu
```

Nên giữ:

- Missing là một bin riêng.
- Outlier trong bin riêng nếu có ý nghĩa.
- Biên cuối là \(-\infty\) và \(+\infty\) để tránh giá trị ngoài range.

---

## 17. PSI cho biến phân loại

Với biến categorical:

- Mỗi category có thể là một bin.
- Category hiếm có thể được gộp vào `OTHER`.
- Missing nên là một nhóm riêng.
- Category mới trong dữ liệu hiện tại nên được đưa vào `UNSEEN` hoặc `OTHER`.

Ví dụ:

```text
Employment Type:
SALARIED
SELF_EMPLOYED
GOVERNMENT
STUDENT
UNEMPLOYED
OTHER
MISSING
UNSEEN
```

Nếu category mới bị bỏ qua, PSI sẽ bị tính sai.

---

## 18. PSI với scorecard dùng WOE

Trong logistic scorecard truyền thống, mỗi biến thường được:

1. Chia bin.
2. Tính WOE.
3. Gán coefficient.
4. Chuyển contribution thành điểm.

Khi giám sát:

- Sử dụng chính bin WOE đã được chốt.
- Tính PSI trên phân phối bin.
- So sánh bad rate trong từng bin.
- Kiểm tra thứ tự monotonic.
- Kiểm tra WOE mới so với WOE ban đầu.
- Kiểm tra coefficient contribution.

Một biến có PSI cao nhưng coefficient nhỏ có thể ít tác động đến score tổng.

Ngược lại, một biến có PSI trung bình nhưng coefficient lớn có thể tác động đáng kể.

Có thể ưu tiên điều tra bằng:

\[
Impact_i \approx PSI_i \times |\beta_i|
\]

Đây không phải công thức chuẩn bắt buộc, nhưng hữu ích để xếp hạng mức độ quan trọng của drift.

---

## 19. PSI theo approved population và through-the-door population

Trong credit risk, cần phân biệt:

### Through-the-door population

Toàn bộ hồ sơ đi vào hệ thống trước quyết định.

### Approved population

Chỉ các hồ sơ được phê duyệt.

### Booked population

Các hồ sơ đã giải ngân hoặc kích hoạt.

PSI có thể khác nhau giữa ba tập này.

Ví dụ:

- Through-the-door PSI thấp.
- Approved PSI cao.

Điều này có thể cho thấy policy hoặc cutoff đã thay đổi.

Nếu chỉ theo dõi approved population, tổ chức có thể bỏ lỡ drift của nhóm bị từ chối.

---

## 20. PSI và reject inference

Khi mô hình chỉ có nhãn default cho khách hàng đã được cấp tín dụng, performance metric bị giới hạn bởi selection bias.

PSI vẫn có thể tính cho toàn bộ population nếu feature và score có sẵn.

Tuy nhiên:

- Score PSI không thay thế reject inference.
- PSI không cho biết nhóm rejected thực sự tốt hay xấu.
- Drift trong approval policy có thể làm performance observed bị sai lệch.

---

## 21. PSI và tính mùa vụ

Một số biến có mùa vụ:

- Thu nhập.
- Chi tiêu.
- Số khoản vay.
- Nhu cầu vay tiêu dùng.
- Sản phẩm tín dụng theo dịp lễ.
- Vay học phí.
- Vay mua nhà.
- Vay nông nghiệp.

Nếu luôn so tháng hiện tại với một tháng cố định, PSI có thể cao do mùa vụ.

Giải pháp:

- So với cùng kỳ năm trước.
- Dùng baseline theo mùa.
- Dùng rolling baseline.
- Tách segment.
- Đặt ngưỡng theo từng tháng hoặc quý.

---

## 22. PSI và sample size

PSI trên mẫu nhỏ có thể không ổn định.

Các vấn đề:

- Một vài record có thể làm thay đổi tỷ lệ đáng kể.
- Bin hiếm có thể có tỷ lệ bằng 0.
- PSI dao động mạnh giữa các kỳ.

Khuyến nghị:

- Đặt minimum sample size.
- Gộp kỳ dữ liệu.
- Dùng rolling window.
- Báo confidence interval.
- Không kết luận chỉ từ một kỳ có volume thấp.

---

## 23. PSI trong dashboard giám sát

Một dashboard PSI nên có:

### Tổng quan

- Score PSI hiện tại.
- Score PSI theo tháng.
- Số biến vượt 0.10.
- Số biến vượt 0.25.
- Top 10 biến có PSI cao nhất.
- Cảnh báo theo segment.

### Chi tiết biến

- PSI tổng.
- PSI contribution theo bin.
- Expected %.
- Actual %.
- Difference.
- Missing rate.
- Mean, median.
- Min, max.
- Category mới.
- Bad rate theo bin.
- WOE theo bin.

### Trend

- PSI theo thời gian.
- Population volume.
- Approval rate.
- Bad rate.
- AUC.
- KS.
- Gini.
- Calibration.

---

## 24. Ví dụ code Python

```python
import numpy as np
import pandas as pd


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins=10,
    epsilon: float = 1e-6
) -> tuple[float, pd.DataFrame]:
    expected = expected.dropna()
    actual = actual.dropna()

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = expected.quantile(quantiles).to_numpy()

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    expected_bucket = pd.cut(
        expected,
        bins=breakpoints,
        include_lowest=True
    )

    actual_bucket = pd.cut(
        actual,
        bins=breakpoints,
        include_lowest=True
    )

    expected_pct = (
        expected_bucket.value_counts(sort=False, normalize=True)
    )

    actual_pct = (
        actual_bucket.value_counts(sort=False, normalize=True)
    )

    expected_pct = expected_pct.clip(lower=epsilon)
    actual_pct = actual_pct.clip(lower=epsilon)

    psi_by_bin = (
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    result = pd.DataFrame({
        "expected_pct": expected_pct,
        "actual_pct": actual_pct,
        "difference": actual_pct - expected_pct,
        "psi_contribution": psi_by_bin
    })

    return float(psi_by_bin.sum()), result
```

### Sử dụng

```python
psi_value, psi_table = calculate_psi(
    expected=train_df["score"],
    actual=current_df["score"],
    bins=10
)

print("PSI:", psi_value)
print(psi_table)
```

---

## 25. Ví dụ code cho biến categorical

```python
import numpy as np
import pandas as pd


def categorical_psi(
    expected: pd.Series,
    actual: pd.Series,
    epsilon: float = 1e-6
) -> tuple[float, pd.DataFrame]:
    expected = expected.fillna("MISSING").astype(str)
    actual = actual.fillna("MISSING").astype(str)

    categories = sorted(set(expected.unique()) | set(actual.unique()))

    expected_pct = (
        expected.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .clip(lower=epsilon)
    )

    actual_pct = (
        actual.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .clip(lower=epsilon)
    )

    psi_by_category = (
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    result = pd.DataFrame({
        "expected_pct": expected_pct,
        "actual_pct": actual_pct,
        "difference": actual_pct - expected_pct,
        "psi_contribution": psi_by_category
    })

    return float(psi_by_category.sum()), result
```

---

## 26. Ví dụ SQL khái niệm

```sql
WITH expected_dist AS (
    SELECT
        score_bin,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS expected_pct
    FROM reference_population
    GROUP BY score_bin
),
actual_dist AS (
    SELECT
        score_bin,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS actual_pct
    FROM current_population
    GROUP BY score_bin
),
joined AS (
    SELECT
        COALESCE(e.score_bin, a.score_bin) AS score_bin,
        GREATEST(COALESCE(e.expected_pct, 0), 0.000001) AS expected_pct,
        GREATEST(COALESCE(a.actual_pct, 0), 0.000001) AS actual_pct
    FROM expected_dist e
    FULL OUTER JOIN actual_dist a
        ON e.score_bin = a.score_bin
)
SELECT
    SUM(
        (actual_pct - expected_pct)
        * LN(actual_pct / expected_pct)
    ) AS psi
FROM joined;
```

---

## 27. Các lỗi triển khai phổ biến

### Tạo bin lại trên dữ liệu actual

Sai vì bin không còn cố định.

### Bỏ qua missing

Sai vì thay đổi missing rate có thể là tín hiệu drift quan trọng.

### Bỏ qua category mới

Sai vì phân phối thực tế không được phản ánh đầy đủ.

### So sánh sample khác định nghĩa

Ví dụ reference là approved population nhưng actual là through-the-door population.

### Dùng baseline quá cũ

PSI cao có thể chỉ phản ánh quá trình kinh doanh đã thay đổi lâu dài.

### Không kiểm tra volume

PSI cao trên mẫu nhỏ có thể không đáng tin cậy.

### Chỉ nhìn PSI tổng

Cần xem contribution theo bin để biết drift nằm ở đâu.

### Retrain ngay khi PSI vượt 0.25

PSI cao chưa đủ để kết luận mô hình cần retrain.

---

## 28. Checklist giám sát PSI

### Trước khi tính

- [ ] Reference population đã được xác định rõ.
- [ ] Actual population có cùng định nghĩa.
- [ ] Bin được xây trên reference.
- [ ] Biên bin được lưu cố định.
- [ ] Missing được xử lý nhất quán.
- [ ] Category mới được xử lý.
- [ ] Epsilon được chuẩn hóa.
- [ ] Sample size đạt yêu cầu.
- [ ] Khoảng thời gian so sánh phù hợp.

### Sau khi tính

- [ ] Xem PSI tổng.
- [ ] Xem contribution theo bin.
- [ ] Xem score PSI.
- [ ] Xem variable PSI.
- [ ] Xem PSI theo segment.
- [ ] Kiểm tra missing rate.
- [ ] Kiểm tra data pipeline.
- [ ] So sánh AUC, Gini và KS.
- [ ] Kiểm tra bad rate.
- [ ] Kiểm tra approval rate.
- [ ] Kiểm tra calibration.
- [ ] Ghi nhận nguyên nhân và hành động.

---

## 29. Mẫu bảng báo cáo PSI

| Variable | PSI | Status | Largest-shift bin | Expected % | Actual % | Possible cause | Action |
|---|---:|---|---|---:|---:|---|---|
| Model Score | 0.08 | Green | 600–650 | 10.1% | 12.4% | Normal variation | Monitor |
| Income | 0.14 | Amber | < 5M | 8.2% | 15.6% | New acquisition channel | Segment analysis |
| Loan Amount | 0.31 | Red | > 200M | 5.0% | 16.8% | Product policy change | Immediate review |
| Employment Type | 0.06 | Green | Self-employed | 22.0% | 25.1% | Seasonal variation | Monitor |
| DTI | 0.27 | Red | > 60% | 4.5% | 13.2% | Data or policy shift | Validate pipeline |

---

## 30. Mẫu kết luận báo cáo

> Score PSI trong kỳ đạt 0.18, cho thấy phân phối điểm đã thay đổi ở mức trung bình so với tập tham chiếu. Drift tập trung chủ yếu ở các score band rủi ro cao. Ba biến có PSI lớn nhất là Loan Amount, Debt-to-Income và Income. AUC giảm nhẹ từ 0.74 xuống 0.72, trong khi bad rate tăng từ 6.1% lên 7.4%. Khuyến nghị kiểm tra thay đổi trong chính sách sản phẩm, kênh acquisition và pipeline của biến Loan Amount trước khi xem xét recalibration hoặc retraining mô hình.

---

## 31. Kết luận

PSI là một chỉ số quan trọng để theo dõi độ ổn định của population trong credit scoring.

PSI phù hợp để trả lời câu hỏi:

> Phân phối dữ liệu hiện tại đã thay đổi bao nhiêu so với giai đoạn tham chiếu?

Tuy nhiên, PSI không trả lời trực tiếp:

> Mô hình hiện tại còn dự báo tốt hay không?

Vì vậy, PSI phải được sử dụng cùng với:

- AUC.
- Gini.
- KS.
- Calibration.
- Bad rate.
- Approval rate.
- Data quality metrics.
- Business performance metrics.

Một quy trình giám sát tốt không chỉ cảnh báo PSI cao, mà còn phải xác định:

1. Drift xuất hiện ở đâu.
2. Drift bắt đầu từ khi nào.
3. Drift đến từ dữ liệu, policy hay thị trường.
4. Drift có ảnh hưởng đến performance hay không.
5. Hành động phù hợp là theo dõi, sửa dữ liệu, recalibrate hay retrain.


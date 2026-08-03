# HCDR EDA — Data insights và findings

> **Phạm vi:** artifact trong `outputs/hcdr/eda/`
>
> **Snapshot dữ liệu:** tải ngày 31/07/2026
>
> **Lần chạy detailed EDA:** 03/08/2026, full data, không sampling
>
> **Event:** `TARGET = 1` nghĩa là gặp khó khăn thanh toán theo định nghĩa của competition

## Kết luận ngắn

**Verified:** Tập train có 307.511 đơn vay, trong đó 24.825 đơn có `TARGET = 1`, tương
đương bad rate 8,0729%. Tín hiệu mô tả mạnh nhất tập trung ở ba biến `EXT_SOURCE_*`;
chúng chiếm 71,46% tổng impurity importance của Random Forest tham chiếu. Tuy nhiên,
`EXT_SOURCE_1` cũng có chênh lệch missing lớn nhất giữa train và test: 56,38% so với
42,12%.

**Finding chính:** mô hình cần xử lý đồng thời ba vấn đề: 

- lệch lớp, missing mang ngữ
  nghĩa nghiệp vụ và category/sentinel đại diện cho trạng thái quy trình.
- Các bad ratetheo nghề nghiệp, giáo dục hoặc hoàn cảnh sống chỉ là liên hệ mô tả
- Không được diễn giải thành quan hệ nhân quả hoặc dùng làm cơ sở cho quyết định tín dụng production.

## Phạm vi và provenance

Batch detailed EDA đọc đủ tám bảng, tổng cộng 58.489.893 dòng. `manifest.json` ghi 66
artifact do runner sở hữu, gồm 39 biểu đồ PNG, các bảng CSV và `summary.md`; toàn bộ 66
đường dẫn đều tồn tại tại thời điểm kiểm tra. Hash của source notebook và
`datasets/raw/home-credit-default-risk/source.json` khớp với manifest.

Trong cùng thư mục còn có `anomaly_findings.csv`, `categorical_bad_rates.csv` và
`column_profile.csv` do pipeline HCDR sinh ở lần chạy khác. Ba file này không thuộc
danh sách 66 artifact của batch notebook. Báo cáo dùng batch detailed EDA làm nguồn
chính và chỉ xem nhóm pipeline là evidence bổ sung; không coi tất cả file trong thư
mục là một lần chạy nguyên khối.

## Findings đã kiểm chứng

### 1. Target lệch lớp rõ rệt

| TARGET | Số đơn |  Tỷ lệ |
| -----: | --------: | -------: |
|      0 |   282.686 | 91,9271% |
|      1 |    24.825 |  8,0729% |

Accuracy của mô hình luôn dự đoán `0` đã gần 91,93%, nên accuracy không phải metric
đủ thông tin. Việc đánh giá nên ưu tiên ROC-AUC như competition, đồng thời bổ sung
PR-AUC, recall/precision theo cutoff, calibration và bad rate theo nhóm score.
Resampling hoặc class weight, nếu dùng, phải chỉ fit trên train split.

### 2. `EXT_SOURCE_*` là cụm tín hiệu nổi bật nhưng có rủi ro missing shift

Ba tương quan Pearson có độ lớn cao nhất với target là `EXT_SOURCE_3` (-0,1789),
`EXT_SOURCE_2` (-0,1605) và `EXT_SOURCE_1` (-0,1553). Random Forest tham chiếu cũng
xếp ba biến này đầu tiên, với importance lần lượt 0,3040; 0,3389; 0,0718. Tổng ba biến
là 71,46% và top 10 feature là 85,12% tổng importance.

Độ lớn tương quan riêng lẻ đều dưới 0,18, vì vậy không có một biến tuyến tính nào tự
nó giải thích target. Đồng thời, missing của `EXT_SOURCE_1` giảm từ 56,38% ở train
xuống 42,12% ở test, chênh 14,26 điểm phần trăm. `EXT_SOURCE_3` chênh 2,04 điểm; median
chênh lệch tuyệt đối của các cột có missing ở cả hai tập chỉ 0,27 điểm. Đây là dấu
hiệu cần kiểm tra distribution shift và calibration riêng theo trạng thái missing,
không phải bằng chứng rằng test “tốt hơn” train.

Random Forest này chỉ phục vụ EDA: 50 cây, depth 8, fit trên toàn bộ
`application_train`, category được mã hóa bằng integer code. Importance chưa qua
cross-validation, có bias theo cardinality và không nên được dùng một mình để chọn
feature.

### 3. Category tạo phân khúc bad rate rõ, nhưng dễ trở thành proxy

Để tránh nhấn mạnh nhóm quá nhỏ, bảng sau chỉ lấy category chiếm ít nhất 1% train:

| Feature                 | Nhóm bad rate cao               | Nhóm bad rate thấp             |  Chênh lệch |
| ----------------------- | -------------------------------- | -------------------------------- | ------------: |
| `OCCUPATION_TYPE`     | Drivers: 11,33% (18.603)         | Accountants: 4,83% (9.813)       | 6,50 điểm % |
| `NAME_EDUCATION_TYPE` | Lower secondary: 10,93% (3.816)  | Higher education: 5,36% (74.863) | 5,57 điểm % |
| `NAME_HOUSING_TYPE`   | Rented apartment: 12,31% (4.881) | House/apartment: 7,80% (272.868) | 4,52 điểm % |
| `NAME_INCOME_TYPE`    | Working: 9,59% (158.774)         | Pensioner: 5,39% (55.362)        | 4,20 điểm % |

Các chênh lệch này hữu ích cho kiểm tra segmentation và interaction, nhưng nghề nghiệp, giáo dục, nhà ở, giới tính và tình trạng gia đình có thể là feature nhạy cảm
hoặc proxy.

Cần đo support, uncertainty, stability theo split và fairness impact trước mọi use case ngoài benchmark. Không suy ra rằng category có bad rate thấp là nguyên nhân làm giảm rủi ro.

### 4. Missing phần lớn có cấu trúc, không phải lỗi ngẫu nhiên đơn giản

Trong application train, `COMMONAREA_AVG/MEDI/MODE` thiếu 69,87%; nhiều feature về tòa nhà khác cũng thiếu trên 65%. Ba tỷ lệ cao nhất của test tương ứng là 68,72%, khá gần train. Trong `previous_application`, hai lãi suất `RATE_INTEREST_*` thiếu 99,64%,
còn `AMT_DOWN_PAYMENT` và `RATE_DOWN_PAYMENT` thiếu 53,64%.

**Inferred:** missing ở các nhóm này có thể phụ thuộc loại sản phẩm, lịch sử quan hệ
hoặc việc khách hàng có tài sản/hợp đồng tương ứng hay không. Vì vậy cần giữ missing
indicator, impute trong train split và kiểm tra bad rate của nhóm missing; không nên
xóa cột chỉ dựa trên một ngưỡng missing toàn cục.

### 5. Sentinel và tail cần được xử lý có provenance

- `DAYS_EMPLOYED = 365243` xuất hiện ở 55.374 hồ sơ (18,01%), bad rate 5,40%. Giá trị
  này làm mean dương 63.815 ngày dù các quantile thông thường là số âm. Cách xử lý
  phù hợp là tạo cờ anomaly rồi thay sentinel bằng missing.
- `CODE_GENDER = XNA` có 4 dòng và `NAME_FAMILY_STATUS = Unknown` có 2 dòng. Support
  quá nhỏ để diễn giải bad rate; vẫn nên giữ cờ trước khi chuẩn hóa thành missing.
- `AMT_INCOME_TOTAL` có median 147.150, p99 472.500 nhưng max 117.000.000. Đây là
  heavy tail rất lớn; scale, log transform hoặc clipping phải học ngưỡng từ train,
  không dùng toàn bộ dữ liệu.

Trong `previous_application`, category placeholder chiếm tỷ trọng lớn: `XAP/XNA`
chiếm 95,83% ở `NAME_CASH_LOAN_PURPOSE`; `XNA` chiếm 63,68% ở
`NAME_PRODUCT_TYPE`, 56,93% ở `NAME_GOODS_CATEGORY` và 51,23% ở
`NAME_SELLER_INDUSTRY`. Chúng có thể mã hóa trạng thái “không áp dụng” hoặc nhánh quy
trình, nên không được thay bằng missing một cách cơ học trước khi kiểm tra theo
`NAME_CONTRACT_STATUS` và loại sản phẩm.

### 6. Nhiều feature nhà ở gần trùng thông tin

Heatmap cho thấy các biến cùng khái niệm với hậu tố `_AVG`, `_MEDI`, `_MODE` có tương
quan rất cao, đặc biệt `FLOORSMAX`, `FLOORSMIN`, `ELEVATORS` và `LIVINGAREA`. Với mô
hình tuyến tính, nên kiểm tra multicollinearity, regularization hoặc đại diện rút gọn.
Với tree model, vẫn cần theo dõi importance bị chia nhỏ giữa các biến tương đương.

## Hành động đề xuất

1. Dùng stratified split và thêm PR-AUC, calibration, cutoff metrics bên cạnh ROC-AUC;
   không dùng accuracy làm tiêu chí chính.
2. Tạo missing indicator cho `EXT_SOURCE_*`, nhóm nhà ở và các feature lịch sử; báo
   metric riêng theo missing state và so sánh train/valid/test.
3. Giữ mapping sentinel rõ ràng (`365243`, `XNA`, `XAP`, `Unknown`) theo từng cột;
   không gom tất cả thành một category chung.
4. Thực hiện feature transform, rare-category grouping, clipping và imputation chỉ
   trên train split, rồi freeze mapping cho valid/test.
5. So sánh ablation `application only` với các block lịch sử và kiểm tra stability,
   thay vì xem impurity importance là bằng chứng đủ về giá trị feature.

## Giới hạn và câu hỏi mở

- EDA là mô tả association, không kiểm chứng causal effect, fairness hay tính phù hợp
  production.
- Dataset không cung cấp một trục thời gian để tạo out-of-time split chuẩn; split
  ngẫu nhiên phân tầng không đo được drift theo thời gian.
- Batch notebook không tính confidence interval cho bad rate category và không đánh
  giá model performance. Những so sánh nhóm nhỏ cần uncertainty trước khi hành động.
- Missing-rate shift mới chỉ là kiểm tra biên; cần PSI/KS hoặc kiểm tra distribution
  theo feature để kết luận drift.

## Nguồn và khả năng tái lập

- Provenance và môi trường: `outputs/hcdr/eda/manifest.json`.
- Tóm tắt lần chạy: `outputs/hcdr/eda/summary.md`.
- Target và category: `outputs/hcdr/eda/application/target_distribution.csv` và
  `application/categorical_bad_rates.csv`.
- Tương quan và importance: `outputs/hcdr/eda/application/target_correlations.csv` và
  `feature_importance/random_forest.csv`.
- Missingness: `outputs/hcdr/eda/missing/*.csv`.
- Phân phối application trước đây:
  `outputs/hcdr/eda/previous_application/categorical_distributions.csv`.
- Numeric tail và anomaly bổ sung: `outputs/hcdr/eda/application/numeric_summary.csv`
  và `outputs/hcdr/eda/anomaly_findings.csv`.
- Runner: `scripts/pipelines/run_hcdr_notebook_eda.py`.

Lệnh tái lập từ repository root:

```bash
uv run python scripts/pipelines/run_hcdr_notebook_eda.py
```

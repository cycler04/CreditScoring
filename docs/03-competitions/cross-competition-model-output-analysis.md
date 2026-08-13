
# HCDR vs HCMS — vì sao hai competition cho ra cùng một hình dạng kết quả?

> **Câu hỏi:** so sánh output của hai benchmark Home Credit Default Risk (HCDR) và
> Home Credit — Credit Risk Model Stability (HCMS): điểm số của chúng “giống nhau” ở
> chỗ nào, và đặc điểm dataset nào tạo ra hành vi đó?
>
> **Phạm vi:** artifact local Stage C của cả hai pipeline. Báo cáo so sánh benchmark
> offline; không suy diễn chất lượng ra quyết định tín dụng thực tế và không mô tả
> solution hạng 1 của competition. Chi tiết từng benchmark nằm ở
> [phân tích output HCDR](home-credit-default-risk/model_output_analysis.md) và
> [phân tích output HCMS](home-credit-model-stability/model_output_analysis.md).

## Trả lời ngắn

**Hai competition không có cùng mức điểm. Chúng có cùng *hình dạng* kết quả.**

Phân biệt này quan trọng vì nó đổi hoàn toàn kết luận:

| Đại lượng | HCDR | HCMS | Nhận xét |
| --- | ---: | ---: | --- |
| AUC test tốt nhất | 0.784172 (blend 3 booster) | 0.830983 (LightGBM) | **Khác** `0.046811` |
| Chênh AUC trong tầng booster | 0.003710 | 0.004596 | **Gần như bằng nhau** (tỷ lệ 1,24×) |
| Chênh AUC toàn bộ 15 model | 0.045383 | 0.163261 | **Khác rõ** (tỷ lệ 3,60×) |
| Spearman thứ hạng 15 model chung | — | — | **0.857143** |

Ba phát biểu **Verified** rút ra từ bảng trên:

1. **Cái giống nhau là mức hội tụ của nhóm boosted tree.** Ở cả hai dataset, bốn
   booster (LightGBM, XGBoost, CatBoost, HistGradientBoosting) nằm gọn trong dải
   `0.0037`–`0.0046` AUC. Con số này gần như không đổi dù dataset khác hẳn nhau về
   kích thước, bad rate, trục thời gian và loại tín hiệu.
2. **Cái giống nhau thứ hai là thứ tự tầng.** Xếp hạng 15 model chung giữa hai
   competition có Spearman `0.857143`: booster/blend trên cùng, bagging ở giữa,
   model bị ràng buộc representation ở dưới.
3. **Cái khác nhau là độ rộng của thang.** HCMS trải rộng gấp 3,6 lần vì baseline
   tuyến tính sụp đổ ở đó (`0.667722`) trong khi ở HCDR nó gần như bám sát booster
   (`0.765819`).

Nguyên nhân **Inferred** chính: hai dataset chia sẻ *cùng một chế độ bài toán* —
tabular đã aggregate, nhị phân, lệch lớp, tín hiệu tập trung vào vài biến số đơn
điệu — nên mọi thuật toán học được ranking tương tự đều chạm cùng một trần. Nhưng
chúng **khác nhau ở dạng của tín hiệu mạnh nhất**: HCDR có sẵn ba external score đã
chuẩn hóa; HCMS chỉ có aggregate thô của lịch sử DPD.

## Cảnh báo trước khi đọc bất kỳ so sánh nào: điểm Kaggle của hai bên không cùng độ tin cậy

Đây là finding quan trọng nhất khi đọc chéo hai run, và nó phải nằm trước mọi bảng số.

**HCDR — submission hợp lệ, số Kaggle dùng được:**

| Model | Local test AUC | Kaggle public | Chênh | Kaggle private | Chênh |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.782425 | 0.77517 | −0.00726 | 0.77228 | −0.01015 |
| LightGBM | 0.780945 | 0.77191 | −0.00903 | 0.77019 | −0.01075 |
| Logistic raw | 0.765819 | 0.75848 | −0.00734 | 0.75117 | −0.01465 |
| WoE scorecard | 0.745614 | 0.73920 | −0.00641 | 0.72919 | −0.01642 |

Bốn submission `COMPLETE`, chấm trên 48.744 dòng test thật. Thứ tự trên leaderboard
**trùng khớp hoàn toàn** thứ tự local, và độ lệch local→public ổn định trong dải
`−0.0064`…`−0.0090`. Đây là **Verified evidence** rằng protocol local HCDR ước lượng
đúng hướng và gần đúng độ lớn.

**HCMS — submission không cho số dùng được:**

| Đường submission | Kết quả | Ghi chú |
| --- | --- | --- |
| Upload 4 file local (`lightgbm`, `xgboost`, `logistic_woe`, `logistic_raw`) | `REJECTED_HTTP_400` | Deadline 27/05/2024 đã qua; competition yêu cầu code submission qua kernel. Probe code-submission trả `HTTP 403`. |
| Kernel version 2 | `FORMAT_ERROR` | Fixture remote sinh `case_id` tổng hợp thay vì ID của public sample. |
| Kernel version 3 | `COMPLETE`, public `0.49961`, private `0.39951` | Notebook chạy đúng mount thật, schema/ID/row-order/bounds đều khớp; local OOT AUC cùng run là `0.821184`. |

**Verified:** public test snapshot local của HCMS chỉ có **10 dòng**
([dataset_inventory.csv](../../outputs/hcms/eda/dataset_inventory.csv)), và
`submission_rows = 10` trong
[validation_metrics.json](../../outputs/hcms/kaggle_notebook_run/v3/validation_metrics.json).
Một AUC `0.49961` là mức ngẫu nhiên; `0.39951` còn dưới ngẫu nhiên.

**Kết luận đúng:** con số Kaggle của HCMS đo đường submission sau deadline trên một
fixture 10 dòng, **không** đo chất lượng model. Nó không mâu thuẫn với local OOT
`0.830983`, vì hai số không đo cùng một thứ. **Unknown:** không thể biết model này sẽ
đạt bao nhiêu trên hidden test thật.

**Hệ quả cho toàn bộ báo cáo này:** mọi so sánh chéo dưới đây dùng **metric local trên
held-out test** của cả hai bên, không dùng leaderboard. Với HCDR ta có một điểm neo
ngoại vi (lệch ~`0.008`); với HCMS ta không có.

## So sánh song song — điểm số thực sự nói gì

### 1. Bảng đối chiếu 15 model chung

| Model | HCDR AUC test | Hạng HCDR | HCMS AUC test | Hạng HCMS | Chênh |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM + XGBoost + CatBoost | 0.784172 | 1 | 0.830632 | 3 | +0.046460 |
| Boosting ensemble | 0.783686 | 2 | 0.830825 | 2 | +0.047140 |
| LightGBM + CatBoost | 0.783632 | 3 | 0.830391 | 4 | +0.046759 |
| CatBoost | 0.782643 | 4 | 0.826199 | 7 | +0.043556 |
| XGBoost | 0.782425 | 5 | 0.829274 | 6 | +0.046849 |
| LightGBM | 0.780945 | 6 | 0.830990 | 1 | +0.050045 |
| HistGradientBoosting | 0.778933 | 7 | 0.829881 | 5 | +0.050948 |
| All-tree ensemble | 0.775995 | 8 | 0.813377 | 8 | +0.037382 |
| LightGBM + CatBoost + Extra Trees | 0.773361 | 9 | 0.787630 | 10 | +0.014269 |
| Logistic raw | 0.765819 | 10 | 0.667722 | **15** | **−0.098097** |
| Random Forest | 0.755405 | 11 | 0.804711 | 9 | +0.049306 |
| Monotonic LightGBM | 0.747242 | 12 | 0.789966 | 11 | +0.042724 |
| WoE scorecard | 0.745614 | 13 | 0.783872 | 12 | +0.038258 |
| GAM | 0.740334 | 14 | 0.707505 | 14 | −0.032828 |
| Extra Trees | 0.738789 | 15 | 0.688772 | 13 | −0.050017 |

Nguồn: [HCDR metrics_C.csv](../../outputs/hcdr/models/metrics_C.csv),
[HCMS metrics.csv](../../outputs/hcms/models/metrics.csv), cùng hai file
`interpretable_metrics.csv`. **Chú ý:** HCDR chấm trên stratified random test 61.503
dòng; HCMS chấm trên out-of-time test 203.345 dòng (tuần 73–91). Chênh cột cuối vì thế
**không** phải hiệu chỉnh chất lượng thuần túy giữa hai model — nó gộp cả khác biệt
population và khác biệt protocol.

### 2. Ba mẫu hình lặp lại ở cả hai competition (**Verified**)

**a. Tầng booster luôn nén chặt.** HCDR `0.003710`, HCMS `0.004596`. Trong cả hai
trường hợp, khoảng cách này nhỏ hơn nhiều so với bất kỳ hiệu ứng nào khác đo được
trong cùng run.

**b. Feature engineering vượt xa lựa chọn thuật toán.**

| Competition | Tăng AUC khi đi Stage A → C | Chênh AUC toàn tầng booster | Tỷ lệ |
| --- | ---: | ---: | ---: |
| HCDR (LightGBM) | +0.016331 | 0.003710 | 4,4× |
| HCMS (LightGBM) | +0.081426 | 0.004596 | 17,7× |

Ở HCDR, cả bốn family model (Logistic, LightGBM, XGBoost, WoE) cùng tăng `0.0118`–
`0.0164` khi đi A→C. Ở HCMS, riêng LightGBM tăng `0.081426` AUC và `0.164088`
Stability. Kết luận chung: **đổi booster gần như không đáng kể so với thêm thông tin.**

**c. Diversity không kèm chất lượng thì làm hỏng blend.** Cùng một hiện tượng, cùng
một thủ phạm:

| Blend | HCDR AUC | HCMS AUC | Brier HCDR | Brier HCMS |
| --- | ---: | ---: | ---: | ---: |
| LightGBM + CatBoost | 0.783632 | 0.830391 | 0.066182 | 0.020387 |
| + Extra Trees | 0.773361 | 0.787630 | 0.080438 | 0.046939 |

Thêm Extra Trees làm giảm AUC `0.010271` (HCDR) và `0.023983` (HCMS), đồng thời làm
Brier xấu đi hơn gấp đôi ở HCMS. Extra Trees ở cả hai dataset có Brier tệ nhất
(`0.195968` và `0.232020`) vì random split cực đoan cộng class weighting đẩy xác suất
xa base rate.

### 3. Ba chỗ hai competition **không** giống nhau (**Verified**)

**a. Ensemble có ích ở HCDR, vô ích ở HCMS.**

| | HCDR | HCMS |
| --- | --- | --- |
| Model đơn tốt nhất | CatBoost 0.782643 | LightGBM 0.830983 |
| Blend tốt nhất | LGB+XGB+Cat 0.784172 | Boosting ensemble 0.830825 |
| Lợi ích của blend | **+0.001529** | **−0.000158** |

Nguyên nhân **Inferred:** Ở HCMS mọi booster GBDT đã hội tụ đến cùng biên giới tín hiệu DPD (đều đạt AUC ~0.826–0.830 trên 244 cột), nên residual của chúng tương quan rất cao và phép equal-weight average không bổ sung thêm thông tin độc lập. Trong khi đó ở HCDR, CatBoost tận dụng tốt thông tin categorical thô nên bổ sung được tính đa dạng cho LightGBM/XGBoost.

**b. Baseline tuyến tính: gần đỉnh ở HCDR, sụp ở HCMS.**

| | HCDR | HCMS |
| --- | ---: | ---: |
| Logistic raw | 0.765819 | 0.667722 |
| Khoảng cách tới model tốt nhất | −0.018353 | −0.163261 |
| WoE scorecard | 0.745614 (21 feature) | 0.783872 (7 feature) |
| WoE − Logistic raw | **−0.020205** | **+0.116150** |

Ở HCDR, binning/WoE *mất* `0.020205` AUC so với logistic thô. Ở HCMS, nó *thu về*
`0.116150` với ít hơn 237 cột. Đây là đảo chiều hoàn toàn, và nó là dấu hiệu rõ nhất
về khác biệt dataset — phân tích ở mục sau.

**c. Random Forest và Extra Trees đổi vị trí tương đối.**

| | HCDR | HCMS |
| --- | ---: | ---: |
| Random Forest, khoảng cách tới đỉnh | −0.028767 | −0.015585 |
| Extra Trees, khoảng cách tới đỉnh | −0.045383 | −0.104403 |

Random Forest ở HCMS bám sát booster hơn (chỉ kém `0.015585`), còn Extra Trees tụt xa
gấp hơn hai lần. **Inferred:** với 1,13 triệu dòng train và tín hiệu tập trung ở vài
biến DPD, bagging đủ dữ liệu để hội tụ tốt; nhưng random split cực đoan của Extra Trees
phá đúng loại threshold tinh tế mà tín hiệu này cần. Bằng chứng gián tiếp: Extra Trees
HCMS xếp **hai missing indicator** vào top-3 importance
([extra_trees.csv](../../outputs/hcms/models/feature_importance/extra_trees.csv)), tức
nó chuyển sang khai thác *có/không có dữ liệu* thay vì giá trị dữ liệu.

## Dataset: chỗ nào giống, chỗ nào khác

### Bảng đối chiếu

| Thuộc tính | HCDR | HCMS | Giống / Khác |
| --- | --- | --- | --- |
| Đơn vị dự đoán | 1 dòng / `SK_ID_CURR` | 1 dòng / `case_id` | **Giống** |
| Target | `TARGET` nhị phân, payment difficulty | `target` nhị phân, payment difficulty | **Giống** |
| Cấu trúc | Quan hệ, one-to-many, cần aggregate trước khi join | Quan hệ, depth 0/1/2 tường minh qua `num_group1/2` | **Giống về bản chất** |
| Dòng train | 307.511 | 1.526.659 | **Khác 5,0×** |
| Dòng vật lý các bảng | 58.489.893 | 243.465.546 | **Khác 4,2×** |
| Số bảng feature | 7 | 16 family + `base` | **Khác** |
| Bad rate | 8,0729% | 3,1437% | **Khác 2,57×** |
| Trục thời gian | **Không có** | `WEEK_NUM` 0–91 | **Khác — quan trọng nhất** |
| Protocol split | Stratified random 60/20/20, seed 42 | Out-of-time nguyên khối tuần 55/18/19 | **Khác** |
| Metric competition | ROC-AUC | Gini stability theo tuần | **Khác** |
| Feature Stage C | 175 raw → 362 transformed | 129 raw → 244 transformed | Khác về số, **giống về cách xây** |
| Categorical native | 16 (CatBoost dùng CTR) | 0 (aggregation quy về số) | **Khác** |
| Tín hiệu mạnh nhất | `EXT_SOURCE_1/2/3` — external score đã chuẩn hóa | DPD aggregates, tuổi, `ROW_COUNT` — thống kê thô | **Khác — quan trọng nhất** |
| Missingness | Có cấu trúc, 50–72% ở nhiều block | Rất thưa; depth 1/2 nhiều case không có record | **Giống về bản chất** |
| Test được chấm | 48.744 dòng, upload file | Hidden, code submission; fixture local 10 dòng | **Khác** |

Nguồn: [cấu trúc HCDR](home-credit-default-risk/home_credit_default_risk_data_structure_report_vi.md),
[cấu trúc HCMS](home-credit-model-stability/home_credit_model_stability_data_structure_report_vi.md),
[EDA HCDR](home-credit-default-risk/data-insights-and-findings.md),
[EDA HCMS](home-credit-model-stability/data-insights-and-findings.md).

### Điểm giống nào tạo ra hội tụ của booster?

Bốn thuộc tính dưới đây **Verified** là chung, và cả bốn đều đẩy các thuật toán về
cùng một nghiệm:

1. **Tabular tĩnh sau aggregate.** Không có chuỗi thời gian thô, ảnh hay văn bản đưa
   vào model. Mọi lịch sử đã bị nén thành `MEAN`/`MAX`/`ROW_COUNT`/`GAP`. Một khi
   thông tin đã ở dạng này, các model học split theo ngưỡng và interaction bậc thấp
   đều tiếp cận được cùng một lượng tín hiệu.
2. **Tín hiệu tập trung vào rất ít biến.** HCDR: ba `EXT_SOURCE_*` chiếm 71,46% impurity
   importance của Random Forest tham chiếu. HCMS: bảy feature qua được bộ lọc WoE đủ để
   đạt `0.783872`, tức 94,3% AUC của model tốt nhất. **Inferred:** khi phần lớn khả
   năng phân tách nằm ở vài biến đơn điệu, không gian hàm mà booster cần khám phá là
   nhỏ, nên các implementation khác nhau đều tìm ra boundary gần giống nhau.
3. **Lệch lớp mạnh.** 8,07% và 3,14%. Số event ít khiến phần đuôi của hàm mất mát bị
   chi phối bởi cùng một nhóm nhỏ hồ sơ, nên các model hội tụ về cùng ranking cho nhóm
   đó.
4. **Cùng harness benchmark.** Cả hai chạy qua cùng kiểu pipeline, cùng seed 42, cùng
   `SimpleImputer(median, add_indicator)`, cùng equal-weight ensemble, cùng bộ model.
   Phần “hình dạng giống nhau” của kết quả có một phần nguyên nhân **là do thiết kế
   benchmark chung**, không hoàn toàn do dữ liệu. Đây là **Verified** và cần nói rõ:
   Spearman `0.857143` không phải bằng chứng thuần túy về dataset.

### Điểm khác nào tạo ra khác biệt về mức và về độ rộng?

**a. Dạng của tín hiệu quyết định số phận baseline tuyến tính.**

`EXT_SOURCE_1/2/3` của HCDR là *normalized external score* — nghĩa là output của một
model khác, đã bị nén về thang liên tục và gần đơn điệu với rủi ro. Một Logistic
Regression trên chúng gần như tối ưu ngay lập tức: `0.765819`, chỉ kém đỉnh `0.018353`.
Binning làm mất độ phân giải của thang đó, nên WoE *tụt* `0.020205`.

HCMS không có cột nào như vậy. Tín hiệu mạnh nhất là
`STATIC_0__avgdpdtolclosure24_3658938P` và
`CREDIT_BUREAU_A_2__pmts_dpd_303P__MEAN` — số ngày quá hạn trung bình, phân phối lệch
nặng với đuôi dài và một khối lớn giá trị 0. Đưa thẳng vào Logistic sau
`StandardScaler` cho `0.667722`. Binning + WoE tuyến tính hóa quan hệ và chặn đuôi,
thu về `+0.116150`.

**Kết luận:** khoảng cách `0.163261` giữa model tốt nhất và kém nhất ở HCMS **không**
phản ánh dataset “khó hơn” cho model mạnh; nó phản ánh dataset **phạt nặng hơn giả
định dạng hàm sai**. Đây là **Inferred** dựa trên hai bằng chứng Verified (đảo chiều
WoE−Logistic, và bản chất `EXT_SOURCE_*` là score đã chuẩn hóa).

**b. Trục thời gian đổi cả protocol lẫn ý nghĩa metric.**

HCDR không có cột thời gian, nên split là stratified random và
[benchmark_protocol.json](../../outputs/hcdr/models/metrics/benchmark_protocol.json)
ghi thẳng `Stability: N/A`. Hệ quả đo được: valid → test gần như đồng nhất, mọi model
chênh `+0.0036`…`+0.0073`.

HCMS có `WEEK_NUM`, nên split là out-of-time và bad rate dịch chuyển thật: train
`0.031254`, valid `0.042548`, test `0.021879`. Hệ quả đo được: gần như mọi model có
**AUC test cao hơn AUC valid** (LightGBM `+0.015912`, WoE `+0.038342`), trong khi
Logistic raw đi ngược `−0.015291`. Cùng LightGBM đó đạt `0.818278` trên random test
so với `0.830984` trên OOT test
([split_protocol_comparison.csv](../../outputs/hcms/models/split_protocol_comparison.csv)).

**Kết luận:** không được so trực tiếp `0.784172` (HCDR) với `0.830983` (HCMS) như hai
mức khó của bài toán. Hai số đến từ hai population và hai protocol khác nhau; chênh
`0.046811` gộp cả hiệu ứng dữ liệu lẫn hiệu ứng cách chia.

**c. Metric Stability của HCMS, trên run này, gần như không thêm thông tin.**

Công thức là `mean(gini) + 88 * min(0, slope) - 0.5 * residual_std`
([stability.py](../../src/home_credit_stability/stability.py)). 11/13 model có slope
dương nên phần phạt drift **bằng 0**; phần còn lại `0.5 * residual_std` chỉ khoảng
`0.019`–`0.026` và gần bằng nhau. Kết quả: xếp hạng theo Stability trùng xếp hạng theo
AUC trừ đúng một cặp hoán vị chênh `0.000081`.

**Verified:** trong run này Stability chỉ phạt Extra Trees (`−0.124921`) và Logistic
raw (`−0.161495`) — hai model vốn đã kém nhất theo AUC. **Kết luận đúng:** HCMS là
competition *về* stability, nhưng artifact hiện tại chưa tạo ra được tình huống mà
stability tách khỏi accuracy.

**d. Khối lượng dữ liệu và độ sâu quan hệ giải thích mức AUC cao hơn.**

HCMS có 5,0× số dòng train và một family depth-2 duy nhất
(`credit_bureau_a_2`, 188,3 triệu dòng) chứa lịch sử DPD chi tiết — đúng nhóm feature
mà mọi model xếp lên đầu. HCDR có lịch sử tương đương ở `bureau_balance` và
`installments_payments`, nhưng tín hiệu mạnh nhất của nó vẫn là external score, không
phải hành vi thô.

**Inferred:** phần lớn chênh lệch `+0.046811` AUC đến từ việc HCMS cho model truy cập
trực tiếp vào hành vi quá hạn ở mức chi tiết, thay vì phải suy ra nó. Chưa có ablation
để xác nhận.

### Calibration: một điểm giống nhau dễ bị bỏ sót

| | HCDR | HCMS |
| --- | ---: | ---: |
| Bad rate test | 0.080728 | 0.021879 |
| Brier của model luôn dự đoán base rate | 0.074211 | 0.021400 |
| Brier tốt nhất đo được | 0.066154 | 0.020465 |
| Giảm tương đối so với baseline | **10,86%** | **4,37%** |

Cả hai đều cho thấy: **AUC cao không kéo theo xác suất tốt.** Ở HCMS, model AUC
`0.830983` chỉ cải thiện Brier 4,37% so với việc đoán base rate cho mọi người. Đây là
hệ quả toán học của event hiếm, không phải lỗi model — nhưng nó có nghĩa là mọi
threshold hay approval policy phải được thiết lập dựa trên calibration đo riêng, không
suy từ AUC. Ở HCMS có sẵn
[cutoffs.csv](../../outputs/hcms/scorecard/cutoffs.csv) và
[approval_bad_rate.png](../../outputs/hcms/scorecard/approval_bad_rate.png) làm điểm
khởi đầu.

## Tổng hợp: trả lời trực tiếp ba câu hỏi

**1. Hai competition có “similar score” không?**

Không ở mức tuyệt đối (`0.784172` vs `0.830983`, chênh `0.046811`). Có ở ba khía cạnh
đo được: mức nén của tầng booster (`0.003710` vs `0.004596`), thứ tự tầng (Spearman
`0.857143`), và hình dạng đường cong hiệu suất theo Stage. Điểm Kaggle thì **không so
được**: HCDR có 4 submission hợp lệ khớp local trong ~`0.008`; HCMS chỉ có một
submission sau deadline chấm trên fixture 10 dòng, trả `0.49961`/`0.39951`.

**2. Vì sao hình dạng giống nhau?**

Ba nguyên nhân, theo mức bằng chứng giảm dần:

- **Verified:** cùng harness benchmark — cùng bộ model, cùng seed, cùng imputer, cùng
  equal-weight ensemble. Một phần của sự giống nhau là do thiết kế, không do dữ liệu.
- **Verified:** cả hai là bài toán tabular nhị phân lệch lớp trên feature đã aggregate,
  với tín hiệu tập trung vào một nhóm rất hẹp (3 biến ở HCDR chiếm 71,46% importance;
  7 biến ở HCMS đạt 94,3% AUC của model tốt nhất).
- **Inferred:** khi không gian hàm cần thiết là nhỏ và đơn điệu, các
  boosted-tree implementation khác nhau hội tụ về cùng boundary; lợi thế kiến trúc
  riêng (leaf-wise vs depth-wise vs symmetric, CTR vs one-hot) không đủ lớn để tách xa.

**3. Đặc điểm dataset nào tạo ra hành vi đó?**

| Đặc điểm | Chung hay riêng | Hệ quả đo được |
| --- | --- | --- |
| Tabular tĩnh, aggregate one-to-many | Chung | Booster nén trong `0.0037`–`0.0046` ở cả hai |
| Tín hiệu tập trung vài biến | Chung | Model 7–21 feature giữ được 94–95% AUC đỉnh |
| Lệch lớp | Chung | Brier chỉ hơn baseline 4,4–10,9%; accuracy vô nghĩa |
| Tín hiệu là external score đã chuẩn hóa | **Riêng HCDR** | Logistic raw chỉ kém đỉnh `0.018353`; WoE *tụt* `0.020205` |
| Tín hiệu là aggregate DPD thô, đuôi dài | **Riêng HCMS** | Logistic raw kém đỉnh `0.163261`; WoE *tăng* `0.116150` |
| Có `WEEK_NUM` và population shift | **Riêng HCMS** | test AUC > valid AUC ở 12/13 model; PSI tuần trung bình `0.207289` |
| 5,0× dòng train, depth-2 DPD chi tiết | **Riêng HCMS** | AUC đỉnh cao hơn `0.046811` |
| Có 16 categorical native | **Riêng HCDR** | CatBoost xếp #4 ở HCDR nhưng #7 ở HCMS |
| Mọi model cùng feature set | **Riêng HCDR** | Blend *tăng* `0.001529`; ở HCMS blend *giảm* `0.000158` |

## Giới hạn và việc cần kiểm chứng tiếp

- **Hai protocol khác nhau khiến so sánh chéo không phải ablation.** Muốn tách hiệu
  ứng dataset khỏi hiệu ứng protocol, cần chạy HCDR với một proxy thời gian (nếu tạo
  được) hoặc chạy HCMS thêm một nhánh stratified random đầy đủ cho mọi model — hiện chỉ
  có LightGBM được chạy cả hai nhánh.
- **Protocol feature của HCMS đã được hoàn thiện đồng nhất** (tất cả các model cây và Logistic raw đều nhận cùng full 244 transformed features). Kết quả khẳng định việc blend không vượt LightGBM ở HCMS là do các booster đã hội tụ về tín hiệu DPD cốt lõi, không phải do bất đồng feature set.
- **Không có repeated CV hay paired significance test ở cả hai bên.** Chênh
  `0.003710` và `0.004596` chưa được gắn nhãn tie hay không tie.
- **Không có ablation nguồn tín hiệu.** Cần rerun bỏ lần lượt `EXT_SOURCE_*` (HCDR) và
  nhóm `pmts_dpd_*` (HCMS) trên cùng folds để định lượng đóng góp.
- **Điểm Kaggle của HCMS vô hiệu.** Nếu cần điểm leaderboard thật, phải chạy code
  submission trong thời hạn hoặc qua late-submission hợp lệ; không suy ra từ local OOT.
- **Stability chưa được kiểm ở chế độ có drift.** Cần kịch bản train-window ngắn hoặc
  tuần shock để metric thể hiện sức phân biệt.
- **Calibration chưa được đánh giá đầy đủ ở cả hai.** Có Brier nhưng chưa có reliability
  curve/calibration slope; FT-Transformer của HCDR còn không export held-out
  probabilities.

## Nguồn local

**HCDR**

- [Bảng benchmark](../../outputs/hcdr/models/metrics/benchmark_table.md),
  [metrics Stage C](../../outputs/hcdr/models/metrics_C.csv),
  [protocol](../../outputs/hcdr/models/metrics/benchmark_protocol.json).
- [Submission scores](../../outputs/hcdr/submissions/submission_scores.csv),
  [run summary](../../outputs/hcdr/run_summary.json).
- [Pipeline](../../src/home_credit_default_rate/pipeline.py),
  [phân tích output](home-credit-default-risk/model_output_analysis.md).

**HCMS**

- [Bảng benchmark](../../outputs/hcms/models/metrics/benchmark_table.md),
  [metrics](../../outputs/hcms/models/metrics.csv),
  [stage metrics](../../outputs/hcms/models/stage_metrics.csv).
- [Stability](../../outputs/hcms/stability/stability_metric.json),
  [gini theo tuần](../../outputs/hcms/stability/gini_by_week.csv),
  [so sánh protocol split](../../outputs/hcms/models/split_protocol_comparison.csv).
- [Submission scores](../../outputs/hcms/submissions/submission_scores.csv),
  [submission attempts](../../outputs/hcms/submissions/submission_attempts.json),
  [validation kernel v3](../../outputs/hcms/kaggle_notebook_run/v3/validation_metrics.json).
- [Pipeline](../../src/home_credit_stability/pipeline.py),
  [stability metric](../../src/home_credit_stability/stability.py),
  [phân tích output](home-credit-model-stability/model_output_analysis.md).

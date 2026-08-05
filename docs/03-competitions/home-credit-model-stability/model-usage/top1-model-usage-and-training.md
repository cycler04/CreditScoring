# Model usage và training evidence của notebook top-1 HCMS local

## Câu trả lời ngắn

Ba notebook local dùng ba model family: ensemble LightGBM, ensemble CatBoost và một
LightAutoML DenseLight model. Chúng chủ yếu là **inference notebooks**: feature schema,
category list và model weights được tải từ Kaggle Dataset bên ngoài. Vì vậy có thể xác
nhận preprocessing và prediction path, nhưng không thể tái dựng đầy đủ training folds,
hyperparameter search hay final blend của winning solution từ repo này.

| Notebook | Artifact tải | Prediction |
| --- | --- | --- |
| LightGBM | nhiều `*.pkl`, `cat_cols.pickle` | trung bình `predict_proba[:,1]` qua models |
| CatBoost | `catboost_model_*`, `cat_cols.pickle` | trung bình models, batch 200.000 |
| LightAutoML | `denselight_model_in_WM.pkl`, saved train/cat/drop columns | `model.predict(...).data`, batch rất nhỏ |

## Training có thể và không thể xác nhận

**Verified:** saved LightGBM cung cấp `feature_name_`; CatBoost cung cấp
`feature_names_`; LightAutoML đi kèm `train_cols/cat_cols/drop_cols`. Test matrix được
select đúng danh sách đó trước inference. Submission lấy official sample làm spine và
ghi `score` theo index `case_id`.

**Unknown từ checkout:** dữ liệu train chính xác, folds, seed, early stopping, sampling,
model count, CV score đáng tin cậy và blend cuối giữa ba family. Import
`StratifiedGroupKFold` hoặc comment LB không phải bằng chứng code đã train theo protocol đó.

## So với pipeline local `src`

Pipeline local thực sự train LightGBM GPU, XGBoost, logistic raw và logistic WoE trên
split tuần. Stage C LightGBM đạt OOT AUC 0,83098 và stability 0,63225 trong artifact.
Kaggle notebook v3 của repo (không phải notebook top-1) dùng 170 feature, GPU, validation
AUC 0,82118; submission `55130117` hoàn tất chấm với public/private 0,49961/0,39951.
Không có official rank vì competition đã kết thúc.

## Kết luận sử dụng

Dùng các notebook top-1 như evidence cho feature/multi-model inference patterns. Muốn
so sánh model công bằng phải có training code/weights provenance, cùng split tuần, cùng
feature snapshot và cùng stability metric; nếu thiếu các điều kiện đó, không gọi chênh
lệch leaderboard là chênh lệch kiến trúc model.

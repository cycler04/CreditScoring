# Khác biệt feature extraction: pipeline `src` và notebook top-1 local

## Kết luận ngắn

Hai bên cùng dùng Polars, base `case_id`, aggregate lịch sử trước left join và đổi date
thành gap. `src` ưu tiên bounded memory, cache và invariant; notebook top-1 tạo moments
và category/recency feature rộng hơn, rồi buộc schema theo saved model. Hướng nên port
trước là calendar feature, dispersion và first/last có ordering rõ; không nên copy
partition handling hoặc fill-null 0 mà chưa kiểm chứng.

| Khía cạnh | `src/home_credit_stability` | Notebook `yuuniekiri` local |
| --- | --- | --- |
| Mục tiêu | benchmark audit được, OOT evaluation | inference competition từ model đã lưu |
| Bộ nhớ | lazy scan, partial Parquet, cache ZSTD | eager read từng file rồi concat |
| Raw selection | tối đa 24 cột/family theo suffix | rộng hơn; contract cuối theo saved model |
| Numeric moments | mean, max | min/max; thêm mean/std/sum/median tùy notebook |
| Category | depth 0 raw; depth 1/2 partition-nunique | first/last/n_unique hoặc min/max tùy script |
| Date | min/max gap | min/max rồi gap; thêm calendar month/weekday |
| Family-specific | không | nhiều rule hard-code cho bureau/person/tax |
| Missing | median + indicator fit train-only | Cat/LAMA fill 0/`Missing`; LGB giữ category/missing |
| Split | tuần sớm/giữa/muộn, lưu membership | training không hiện diện đầy đủ trong scripts |
| Join contract | assert unique + row count | ít assertion; `unique(case_id)` sau concat |
| Provenance | source hash, cache, matrix, artifacts | phụ thuộc external saved-model datasets |

## Khoảng trống feature của `src`

1. Không có `decision_month`/`decision_weekday`.
2. Không có min, std, sum, median numeric ở phần lớn family.
3. Không giữ first/last category hoặc record theo `num_group1`.
4. Không có range/volatility và interaction family-specific.
5. Depth 2 không aggregate hierarchy trước khi về case.

## Thứ tự port đề xuất

1. Thêm calendar feature và test rằng `WEEK_NUM` vẫn chỉ dùng cho protocol.
2. Thêm `min/std/sum` theo partial-statistic merge đúng toán học; đo delta Stage B/C.
3. Thêm first/last sau khi định nghĩa tie-breaking và partition merge.
4. Sửa global `NUNIQUE` thay vì cộng per-partition.
5. Chỉ sau đó thử domain interaction/range hard-code và ablation theo stability.

Mỗi port phải fit/selection trên train weeks, đo OOT AUC/Gini stability, peak memory và
giữ invariant một dòng/case. Leaderboard notebook là nguồn ý tưởng, không phải ground
truth cho pipeline production.


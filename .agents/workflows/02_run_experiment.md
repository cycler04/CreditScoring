# Thiết kế và chạy thử nghiệm R&D

## Khi dùng

Dùng khi cần thử code/paper, benchmark, ablation, kiểm chứng giả thuyết, tái lập
kết quả hoặc đánh giá dataset/converter.

## Quy trình

1. Viết hypothesis, biến thay đổi, baseline, metric và điều kiện dừng.
2. Ghi trạng thái đầu vào: revision/submodule, config, dataset identity, sample
   size và environment.
3. Ước lượng disk, RAM/VRAM và thời gian. Với dataset lớn, kiểm metadata trước.
4. Chạy smoke test nhỏ nhất; kiểm schema/output/log trước khi scale.
5. Chạy baseline và chỉ thay một nhóm biến có chủ đích.
6. Lưu command nguyên vẹn, resolved config, log, metric và artifact path.
7. Kiểm tra failure mode, seed/sample sensitivity và dữ liệu rò rỉ nếu liên quan.
8. Kết luận hypothesis được hỗ trợ, bị bác bỏ hay chưa đủ bằng chứng.

## Điều kiện dừng an toàn

Dừng và báo cáo nếu input không đúng identity, output có schema sai, disk/RAM
tăng ngoài dự kiến, metric không so sánh được, hoặc cần thao tác hệ thống bên
ngoài chưa được ủy quyền.

## Output tối thiểu

- Hypothesis và success criteria.
- Command/config/environment.
- Kết quả có baseline.
- Artifact/log path.
- Failure/limitation.
- Kết luận và next experiment.


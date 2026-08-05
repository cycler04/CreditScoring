# Thiết kế và chạy thử nghiệm R&D

## Khi dùng

Dùng khi cần thử code/paper, benchmark, ablation, kiểm chứng giả thuyết, tái lập
kết quả hoặc đánh giá dataset/converter.

## Quy trình

1. Viết **Why**: vấn đề/failure mode chưa được giải quyết, bằng chứng hiện có,
   tác động của nó và success criteria quan sát được.
2. Viết **How**: hypothesis và ánh xạ từng biến thay đổi, model/training choice,
   baseline, metric hoặc benchmark về phần vấn đề mà nó kiểm tra. Không benchmark
   một metric chỉ vì nó phổ biến nếu metric đó không đo success criteria.
3. Ghi trạng thái đầu vào: revision/submodule, config, dataset identity, sample
   size và environment.
4. Ước lượng disk, RAM/VRAM và thời gian. Với dataset lớn, kiểm metadata trước.
5. Chạy smoke test nhỏ nhất; kiểm schema/output/log trước khi scale.
6. Chạy baseline và chỉ thay một nhóm biến có chủ đích.
7. Lưu command nguyên vẹn, resolved config, log, metric và artifact path.
8. Kiểm tra failure mode, seed/sample sensitivity và dữ liệu rò rỉ nếu liên quan.
9. Kết luận hypothesis được hỗ trợ, bị bác bỏ hay chưa đủ bằng chứng; nói rõ
   vấn đề ban đầu đã được xử lý đến mức nào.

## Điều kiện dừng an toàn

Dừng và báo cáo nếu input không đúng identity, output có schema sai, disk/RAM
tăng ngoài dự kiến, metric không so sánh được, hoặc cần thao tác hệ thống bên
ngoài chưa được ủy quyền.

## Output tối thiểu

- Why: vấn đề, evidence ban đầu và success criteria.
- How: hypothesis cùng problem → experimental choice map.
- Command/config/environment.
- Kết quả có baseline và diễn giải nó trả lời vấn đề ra sao.
- Artifact/log path.
- Failure/limitation.
- Kết luận và next experiment.

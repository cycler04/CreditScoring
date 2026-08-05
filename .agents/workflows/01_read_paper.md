# Đọc và đánh giá paper

## Khi dùng

Dùng khi cần đọc paper/PDF, giải thích phương pháp, đối chiếu nhiều paper, tìm
liên hệ paper-code hoặc tạo research note.

## Quy trình

1. Chốt câu hỏi nghiên cứu và output mong muốn trước khi đọc sâu.
2. Xác minh đúng phiên bản paper, nguồn chính thức và code/dataset đi kèm nếu có.
3. Đọc và tổng hợp theo trục **Why → How**:
   - **Why:** vấn đề cụ thể là gì, ai/hệ thống nào gặp vấn đề, cách trước đó thất
     bại hoặc còn gap ở đâu, hậu quả là gì và tiêu chí nào cho thấy vấn đề đã được xử lý;
   - **How:** phương pháp xử lý vấn đề như thế nào; mỗi thành phần phải chỉ rõ nó
     xử lý pain point, giả định hoặc failure mode nào trong phần Why.
4. Với PDF dài, tìm abstract, problem statement, related work, method, experiments, limitations
   và appendix liên quan trước; không nạp toàn bộ artifact vào context nếu có
   thể trích đúng trang/section.
5. Phân tích phần How theo các nhóm phù hợp với paper, không bắt buộc paper nào cũng có đủ:
   - **Modeling:** biểu diễn/kiến trúc/objective nào được chọn, vì sao lựa chọn đó
     phù hợp với vấn đề và cơ chế kỳ vọng là gì;
   - **Training:** dữ liệu, supervision, loss, sampling, optimization hoặc protocol
     nào khắc phục gap nào; nêu giả định và nguy cơ leakage nếu có;
   - **Benchmark:** mỗi dataset, baseline, metric và ablation kiểm tra câu hỏi nào;
     kết quả có thực sự chứng minh pain point đã được xử lý hay chỉ cải thiện proxy;
   - **Inference/system:** chỉ mô tả khi nó giải quyết constraint như latency,
     memory, robustness, deployment hoặc khả năng mở rộng.
6. Ghi riêng:
   - claim của tác giả;
   - bằng chứng/metric/dataset hỗ trợ;
   - diễn giải của người đọc;
   - giới hạn, missing baseline và threat to validity.
7. Nếu có code, ánh xạ claim quan trọng tới module/config/checkpoint thực tế.
8. Kết thúc bằng kết luận trả lời câu hỏi ban đầu, mức tin cậy và 1–3 thử nghiệm
   tiếp theo có khả năng bác bỏ hoặc củng cố kết luận.

## Output tối thiểu

- Citation/identity của nguồn.
- Câu hỏi nghiên cứu.
- **Why:** vấn đề, gap của cách hiện tại, hậu quả và success criteria.
- **How:** problem → mechanism map, rồi modeling/training/benchmark khi phù hợp.
- Evidence table hoặc danh sách claim → evidence.
- Limitations và điểm chưa rõ.
- Liên hệ với workspace.
- Đề xuất thử nghiệm tiếp theo.

Không chép dài nguyên văn. Quote chỉ dùng khi wording chính xác là đối tượng cần
phân tích.

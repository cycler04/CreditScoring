# Workflow: đóng gói nghiên cứu thành báo cáo Markdown

Mục tiêu là biến việc đọc code, dataset, artifact và paper thành tài liệu có thể truy lại,
không phải bản tóm tắt mất nguồn hoặc một dump ghi chú dài.

Các report đều viết bằng Tiếng Việt, dù prompt là Tiếng Anh

## 1. Xác định câu hỏi và file đích

- Viết một câu hỏi chính và phạm vi không làm.
- Tìm báo cáo hiện có trong `docs/`; cập nhật file đúng chủ đề thay vì tạo bản trùng.
- Dùng `notes/` cho nguồn/quan sát thô. Dùng `docs/` cho kết quả đã tổng hợp.
- Không đưa domain knowledge vào `.agents/memory/`; memory chỉ giữ preference/correction.

## 2. Thu bằng chứng

Ưu tiên theo thứ tự: code/runtime artifact của workspace, documentation/paper gốc, rồi mới
đến nguồn tổng hợp. Với file dài hoặc tìm kiếm rộng, dùng agent mô tả ở
`../agents/02_research_reader.md` để nhận evidence map thay vì nội dung thô.

Mỗi claim quan trọng cần có source URL hoặc local path/command. Ghi ngày truy cập cho thông
tin dễ thay đổi. Không biến inference thành fact.

## 3. Tổng hợp theo Why → How

Cấu trúc mặc định:

1. câu trả lời ngắn hoặc ý tưởng chính;
2. **Why — vấn đề cần xử lý:** bối cảnh, pain point/failure mode, gap của cách
   hiện tại, hậu quả và tiêu chí thành công;
3. **How — cách xử lý:** bản đồ problem → mechanism, sau đó là modeling,
   training, benchmark/evaluation và system/inference khi phù hợp;
4. bằng chứng hoặc kết quả đo và chúng chứng minh phần nào của vấn đề;
5. giới hạn, unknown, bẫy và phần vấn đề còn chưa được xử lý;
6. kết luận và bước kiểm chứng tiếp theo;
7. nguồn.

Trong phần How, không liệt kê component độc lập với mục tiêu. Với mỗi lựa chọn quan trọng,
trả lời đủ ba câu: **nó sửa vấn đề nào, sửa bằng cơ chế gì, evidence nào cho thấy nó có
hiệu quả**. Benchmark phải nêu câu hỏi được kiểm tra, baseline/protocol/metric và liệu kết
quả đo trực tiếp pain point hay chỉ là proxy. Nếu một chi tiết không ánh xạ được về Why và
không cần cho tái lập, lược bỏ nó.

Chỉ thêm bảng khi cần so sánh nhiều field/model/dataset. Không chép parameter không ảnh hưởng
hành vi. Với dataset, luôn ghi action/state semantics, timestamp, modalities và conversion loss; đừng chỉ ghi tên và dung lượng.

Với các diagram mermaid, vẽ các biểu đồ có sử dụng nhiều row, không để 1 row dài trong hình vẽ để dễ nhìn khi render

## 4. Phân biệt trạng thái

Dùng nhãn rõ khi có nguy cơ đọc nhầm:

- **Verified:** đã chạy/đo hoặc được source gốc xác nhận.
- **Inferred:** suy luận có căn cứ, nói rõ căn cứ.
- **Unknown:** chưa có dữ liệu đủ kết luận.
- **Planned:** thiết kế hoặc việc sẽ làm, chưa phải runtime truth.

## 5. Tự kiểm trước khi lưu

- Link/path có resolve không?
- Claim có bằng chứng gần nó không?
- Có lặp sự thật đã nằm ở file khác không?
- Tài liệu có nói quá năng lực code hiện tại không?
- Mỗi phần modeling/training/benchmark có chỉ rõ nó xử lý vấn đề nào không?
- Sáu tháng sau có biết đã chạy lệnh nào, trên input nào và giới hạn gì không?

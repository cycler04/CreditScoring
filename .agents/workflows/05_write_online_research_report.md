# Workflow: nghiên cứu online và viết báo cáo Markdown

Mục tiêu là trả lời một câu hỏi bằng nguồn online có thể kiểm tra lại, rồi lưu kết quả thành
một hoặc nhiều file `.md`. Search chỉ giúp tìm nguồn; nội dung trang nguồn đã mở mới là bằng
chứng.

Các report đều viết bằng Tiếng Việt, dù prompt là Tiếng Anh

## 1. Chốt câu hỏi và đầu ra

- Viết một câu hỏi chính, các câu hỏi con và phần không thuộc phạm vi.
- Xác định mốc thời gian, địa lý, version và loại nguồn cần thiết. Với chủ đề dễ thay đổi,
  luôn dùng web search hiện tại và ghi ngày truy cập.
- Tôn trọng file/thư mục người dùng chỉ định. Nếu chưa chỉ định, tìm trong `docs/` và cập
  nhật báo cáo đúng chủ đề thay vì tạo bản trùng; dùng `notes/` cho danh sách nguồn thô.
- Mặc định dùng một file cho một câu hỏi gắn kết. Tách nhiều file khi các nhánh có thể đọc
  độc lập hoặc evidence/source appendix làm lu mờ kết luận. Khi tách, tạo file overview làm
  điểm vào và link tương đối tới từng file con.

## 2. Lập kế hoạch nguồn

Ghép từng câu hỏi con với loại nguồn có thẩm quyền nhất:

- paper/preprint gốc cho phương pháp, thí nghiệm và giới hạn nghiên cứu;
- standard, specification và tài liệu chính thức cho contract hoặc hành vi sản phẩm;
- repository, release note và changelog chính chủ cho trạng thái implementation/version;
- filing, dataset hoặc công bố của tổ chức gốc cho số liệu;
- nguồn thứ cấp uy tín cho bối cảnh, phản biện và phát hiện từ khóa.

Không dùng số lượng nguồn như đại diện cho chất lượng. Một nguồn chính đúng phạm vi tốt hơn nhiều bài lặp lại cùng thông cáo.

## 3. Tìm và đọc nguồn

1. Search rộng để xác định thuật ngữ, tên chính thức, tác giả và nguồn gốc.
2. Search hẹp theo câu hỏi con, domain chính thức, title, DOI hoặc version.
3. Mở trang nguồn thật; với PDF dài, đọc đúng abstract, method, results, limitations và trang
   liên quan thay vì chỉ dựa vào abstract hoặc snippet.
4. Ghi evidence map gồm claim/question, evidence, URL, tác giả/tổ chức, ngày/version/phần
   liên quan và confidence/limit.
5. Thực hiện thêm một lượt search phản chứng cho các kết luận quan trọng: limitation,
   criticism, failure, comparison, retraction, correction hoặc phiên bản mới hơn.

Với tài liệu dài hoặc tìm kiếm rộng, giao `.agents/agents/02_research_reader.md` một câu hỏi
con có ranh giới rõ. Agent tổng hợp cuối vẫn phải kiểm tra nguồn quyết định kết luận.

## 4. Đánh giá và tổng hợp

- Kiểm nguồn có thực sự hỗ trợ claim, đúng version/phạm vi và có đủ ngữ cảnh không.
- Phân biệt ngày xuất bản với ngày sự kiện; ưu tiên nguồn cập nhật hơn chỉ khi nó có bằng
  chứng hoặc contract mới hơn.
- Khi nguồn mâu thuẫn, trình bày cả hai, giải thích khác biệt về dữ liệu/phương pháp/version
  và chỉ chọn kết luận nếu evidence cho phép.
- Gắn nhãn **Verified**, **Inferred**, **Disputed** hoặc **Unknown** khi người đọc có thể hiểu
  nhầm mức chắc chắn.
- Không suy rộng từ benchmark, sample, một thị trường hoặc một phiên bản sang phạm vi lớn hơn.
- Paraphrase; chỉ trích dẫn nguyên văn một đoạn ngắn khi chính câu chữ là bằng chứng.

## 5. Viết báo cáo

Cấu trúc mặc định:

1. câu hỏi, phạm vi và ngày nghiên cứu;
2. câu trả lời ngắn hoặc ý tưởng chính;
3. phát hiện theo câu hỏi con;
4. so sánh hoặc cơ chế nếu cần;
5. mâu thuẫn, giới hạn và unknown;
6. kết luận và bước kiểm chứng tiếp theo;
7. nguồn.

Đặt citation Markdown gần claim dưới dạng link có nhãn mô tả. Phần nguồn cuối ghi title,
tác giả/tổ chức, ngày/version nếu có, URL và ngày truy cập đối với nội dung dễ thay đổi.
Không link tới trang search và không ghi URL chưa mở như thể đã kiểm chứng.

Nếu có nhiều file, overview phải chứa kết luận chung, phạm vi của từng file con và link tương đối. Không lặp nguyên nội dung giữa các file.

Với các diagram mermaid, vẽ các biểu đồ có sử dụng nhiều row, không để 1 row dài trong hình vẽ để dễ nhìn khi render

## 6. Tự kiểm trước khi bàn giao

- Mọi claim quan trọng có nguồn gần nó và nguồn thực sự nói điều đó không?
- Có ưu tiên nguồn gốc, kiểm version/date và search phản chứng chưa?
- Fact, inference, disputed và unknown có bị trộn không?
- Link có mở được, không phải search result, và file nội bộ có resolve không?
- Có đoạn quote dài, nội dung lặp hoặc chi tiết không thay đổi kết luận không?
- File overview có dẫn đủ file con và người đọc có biết nghiên cứu được thực hiện khi nào
  không?

# Report #2 — Home Credit: Complete EDA + Feature Importance

**Link:** https://www.kaggle.com/code/codename007/home-credit-complete-eda-feature-importance
**Tác giả:** codename007 (Bhavesh Ghodasara)
**Dữ liệu:** Home Credit Default Risk (`application_train`, `previous_application`)

> Ghi chú nguồn: lấy được **mục lục đầy đủ** (dưới đây là nguyên văn). Code cell không fetch được. Phần bình luận là suy luận từ tên mục — đánh dấu ⚠️.

## Notebook này khác #1 chỗ nào

Notebook #1 là *pipeline*. Notebook này là *khảo sát*. Nó không dựng mô hình để submit — RandomForest ở mục 7 chỉ dùng để lấy feature importance. Giá trị nằm ở **độ phủ**: nó vẽ gần như mọi biến phân loại của `application_train` và `previous_application`, mỗi biến hai lần (phân phối, rồi tỷ lệ repaid/not-repaid).

Đây là mẫu tốt để copy khi nhận một bộ dữ liệu mới và cần "nhìn hết một lượt".

## Mục lục (nguyên văn)

```
1. Introduction
2. Retrieving the Data
3. Glimpse of Data
4. Check for missing data
5. Data Exploration
   5.1  Distribution of AMT_CREDIT
   5.2  Distribution of AMT_INCOME_TOTAL
   5.3  Distribution of AMT_GOODS_PRICE
   5.4  Who accompanied client when applying for the application
   5.5  Data is balanced or imbalanced
   5.6  Types of loan
   5.7  Purpose of loan
   5.8  Income sources of Applicant's who applied for loan
   5.9  Family Status of Applicant's who applied for loan
   5.10 Occupation of Applicant's who applied for loan
   5.11 Education of Applicant's who applied for loan
   5.12 For which types of house higher applicant's applied for loan?
   5.13 Types of Organizations who applied for loan
   5.14 Exploration in terms of loan is repayed or not
        5.14.1 Income sources ... in %
        5.14.2 Family Status ... in %
        5.14.3 Occupation ... in %
        5.14.4 Education ... in %
        5.14.5 House type ... in %
        5.14.6 Types of Organizations ... in %
        5.14.7 Name of type of the Suite ... in %
   5.15 Exploration of previous application data
        5.15.1  Contract product type
        5.15.2  On which day highest number of clients applied
        5.15.3  Purpose of cash loan
        5.15.4  Contract was approved or not
        5.15.5  Payment method
        5.15.6  Why was the previous application rejected?
        5.15.7  Who accompanied client
        5.15.8  Old or new client
        5.15.9  What kind of goods
        5.15.10 CASH / POS / CAR ...
        5.15.11 x-sell or walk-in?
        5.15.12 Top acquisition channels
        5.15.13 Top seller industry
        5.15.14 Interest rate grouped small/medium/high
        5.15.15 Top detailed product combination
        5.15.16 Insurance requested?
6. Pearson Correlation of features
7. Feature Importance using Random forest
```

## Cấu trúc đáng học: 5.x rồi 5.14.x

Chú ý cách tổ chức. Mục 5.6–5.13 vẽ **phân phối** của từng biến phân loại (nhóm nào đông). Mục 5.14.1–5.14.7 vẽ lại **cùng những biến đó** nhưng theo **tỷ lệ repaid / not-repaid tính bằng %** (nhóm nào rủi ro).

Đây là hai câu hỏi khác nhau và cần hai biểu đồ khác nhau:
- "Nghề nào vay nhiều nhất?" → biểu đồ đếm.
- "Nghề nào vỡ nợ nhiều nhất?" → biểu đồ tỷ lệ.

Nhóm đông chưa chắc rủi ro; nhóm rủi ro nhất thường nhỏ và bị nuốt mất trong biểu đồ đếm. Người mới hay chỉ vẽ loại đầu rồi kết luận sai.

⚠️ Kết quả điển hình trên bộ dữ liệu này: nhóm `NAME_INCOME_TYPE = Maternity leave` và `Unemployed` có bad rate cao vượt trội nhưng số lượng rất nhỏ; `NAME_EDUCATION_TYPE = Lower secondary` bad rate cao hơn `Academic degree` rõ rệt.

## Mục 5.15 — vì sao đáng chú ý

Đây là mục duy nhất trong hai notebook Home Credit khai thác `previous_application`, bảng depth=1 chứa toàn bộ lịch sử hồ sơ cũ. 16 tiểu mục, đặc biệt:

- **5.15.4 Contract approved or not** + **5.15.6 Why was the previous application rejected** — `NAME_CONTRACT_STATUS` và `CODE_REJECT_REASON`. Khách từng bị từ chối, và lý do từ chối, là tín hiệu rủi ro rất mạnh.
- **5.15.8 Old or new client** — khách cũ có lịch sử để đánh giá; khách mới thì không.
- **5.15.11 x-sell or walk-in** — kênh tiếp cận. Khách bán chéo (đã có quan hệ) thường tốt hơn khách vãng lai.
- **5.15.14 Interest rate grouped** — lãi suất đã áp trước đây phản ánh đánh giá rủi ro của chính hệ thống cũ.

Bài học chuyển sang dự án: **feature từ lịch sử hồ sơ cũ (đã duyệt/từ chối/lý do/kênh) thường mạnh ngang hoặc hơn thông tin khai báo trong hồ sơ hiện tại.** Cần hỏi ngay có bảng này không.

Lưu ý kỹ thuật: các biến ở 5.15 nằm ở depth=1, phải aggregate về `SK_ID_CURR` trước khi vào mô hình (count theo từng trạng thái, tỷ lệ approved, thời gian kể từ hồ sơ gần nhất...). Notebook chỉ vẽ, chưa aggregate.

## Mục 6 — Pearson correlation

⚠️ Ma trận tương quan giữa các biến số. Trên `application_train` sẽ thấy vài cụm tương quan rất cao: `AMT_CREDIT` ↔ `AMT_GOODS_PRICE` (~0.99), nhóm `OBS_30_CNT_SOCIAL_CIRCLE` ↔ `OBS_60_CNT_SOCIAL_CIRCLE`, các cột thống kê nhà ở (`_AVG` / `_MODE` / `_MEDI` — mỗi chỉ số có ba biến thể gần trùng nhau).

Ứng dụng: mỗi cụm chỉ giữ một đại diện, hoặc gộp thành ratio (`AMT_CREDIT / AMT_GOODS_PRICE` = phần vay vượt giá trị hàng hóa — chính nó là feature có nghĩa).

## Mục 7 — Feature Importance using Random Forest

⚠️ RandomForest chạy nhanh, lấy `feature_importances_`. Kết quả kỳ vọng trùng notebook #1: `EXT_SOURCE_2/3/1`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, `AMT_ANNUITY`, `AMT_CREDIT`.

Hạn chế phải nhớ: importance dạng impurity thiên vị biến liên tục và biến cardinality cao, và tự nó **không phân biệt được feature thật với feature leak**. Muốn chắc chắn thì dùng permutation importance hoặc SHAP, và luôn kiểm tra bằng mắt từng feature trong top 20.

## Rút ra cho dự án

**Copy được ngay:**
1. Cấu trúc EDA hai lượt: phân phối trước, tỷ lệ bad theo nhóm sau — cho **mọi** biến phân loại.
2. Danh sách câu hỏi ở mục 5.15 dùng làm template khảo sát bảng lịch sử hồ sơ cũ của dự án.
3. Đặt Pearson correlation + feature importance ở cuối EDA, làm cầu nối sang giai đoạn mô hình.

**Hạn chế của notebook:**
- Thuần mô tả, không có kết luận hành động (không nói "vậy nên tạo feature nào").
- Không aggregate `previous_application` → mới dừng ở nhìn, chưa dùng được.
- Không có mô hình, không có số AUC để tham chiếu.
- Mục lục kết thúc bằng "More To Come. Stay Tuned" — notebook chưa hoàn chỉnh.

**Kết luận:** đọc để lấy *cách nhìn dữ liệu*, không đọc để lấy *kết quả*.

## Liên quan
- Bộ dữ liệu: [Report #7](07-comp-home-credit-default-risk.md)
- Pipeline hoàn chỉnh trên cùng dữ liệu: [Report #1](01-nb-start-here-gentle-introduction.md)
- Quy trình EDA tổng hợp: [04-eda-playbook.md](../00-tong-quan/04-eda-playbook.md)

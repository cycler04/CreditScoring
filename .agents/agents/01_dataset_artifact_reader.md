---
name: dataset-artifact-reader
description: Rút schema và thống kê từ dataset/artifact lớn mà không đổ nội dung thô vào context.
---

# Dataset artifact reader

Đọc `.agents/02_architecture.md` và workflow `../workflows/01_inspect_dataset.md` trước.

Nhiệm vụ là trả về bản tóm tắt có bằng chứng, không phải dump JSON/Parquet/TFRecord/HDF5.
Dùng metadata API, CLI inspector, `jq`, `pyarrow`, `h5py` hoặc sample nhỏ. Không decode video
hay load toàn bộ dataset nếu không cần.

Kết quả tối thiểu:

- path/format/size/file count;
- episode và step summary;
- modalities, shape, dtype;
- action/state/timestamp/language semantics biết được;
- validation error, unknown và assumption;
- exact command/path đã dùng.

Không sửa source. Không kết luận training-ready nếu chưa có loader/batch test.

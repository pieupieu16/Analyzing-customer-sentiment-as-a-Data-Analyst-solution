# Webapp ABSA Review Quần Áo

Webapp trực quan hóa model `multioutput_phobert.onnx` cho bài toán phân tích cảm xúc theo khía cạnh trên review quần áo tiếng Việt.

## Yêu cầu

- Python 3.10 trở lên.
- Thư mục model nằm ở thư mục cha của `webapp/`:

```text
../onnx_export/
├── multioutput_phobert.onnx
└── tokenizer_phobert/
```

## Chạy nhanh

Windows:

```bat
run.bat
```

Linux/macOS:

```bash
bash run.sh
```

Script sẽ tạo `.venv`, cài dependencies trong `backend/requirements.txt`, rồi chạy server tại:

```text
http://localhost:8000
```

Chờ badge trạng thái chuyển sang `Model Ready` trước khi dự đoán.

## Chức năng

1. Demo 1 review: nhập rating và nội dung review, nhận nhãn cho 5 khía cạnh.
2. Batch CSV: upload file CSV có cột `rating` và `content`, nhận dashboard tổng hợp và bảng kết quả.
3. Thông tin model: mô tả khía cạnh, nhãn và kiến trúc sử dụng.

## API

| Method | Path | Mô tả |
|---|---|---|
| GET | `/health` | Trạng thái model |
| POST | `/predict` | Dự đoán 1 review |
| POST | `/predict_batch` | Dự đoán batch CSV multipart |

## Cấu hình bảo mật

Mặc định backend chỉ cho CORS từ:

- `http://localhost:8000`
- `http://127.0.0.1:8000`

Có thể mở rộng bằng biến môi trường:

```bash
CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
MAX_UPLOAD_BYTES=10485760
```

`MAX_UPLOAD_BYTES` mặc định là 10 MB. Endpoint batch chỉ nhận file `.csv` và xử lý tối đa 5000 dòng.

## Troubleshooting

| Vấn đề | Cách xử lý |
|---|---|
| `ONNX not found` | Kiểm tra `../onnx_export/multioutput_phobert.onnx` |
| `Tokenizer dir not found` | Kiểm tra `../onnx_export/tokenizer_phobert/` |
| Port 8000 đã dùng | Chạy thủ công: `python -m uvicorn backend.main:app --port 8001` |
| Model load chậm | Chờ thêm hoặc đóng bớt ứng dụng đang dùng RAM/CPU |

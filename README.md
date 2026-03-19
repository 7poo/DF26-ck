# DataFlow Season 2 Submission

Repository này gồm 2 phần chính:

- `data/`: EDA, training, inference và tạo submission cho bài toán dự đoán.
- `antigravity_system/`: hệ thống demo fullstack gồm Streamlit + FastAPI + Kafka + MinIO + AI Agent.

## Project Title

Team Antigravity - DataFlow Season 2 Submission

## Prerequisites

- Python `3.10+`
- `pip`
- Docker Engine + Docker Compose plugin nếu muốn chạy hệ thống fullstack
- RAM khuyến nghị:
  - `4 GB` cho EDA/training nhẹ trên CPU
  - `8 GB` trở lên để chạy `antigravity_system`
- GPU là tùy chọn. Script training vẫn chạy được trên CPU nhưng sẽ chậm hơn nhiều.

## Project Structure

```text
.
|-- README.md
|-- data/
|   |-- Model.py
|   |-- EDA.py
|   |-- X_train.csv
|   |-- X_val.csv
|   |-- X_test.csv
|   |-- Y_train.csv
|   |-- Y_val.csv
|   `-- v47_snapshot_ep72.pt
`-- antigravity_system/
    |-- README.md
    |-- .env.example
    |-- docker-compose.yml
    |-- backend/
    `-- frontend/
```

## Installation

### Python environment

Windows:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib scikit-learn torch fastapi uvicorn streamlit requests confluent-kafka google-generativeai python-dotenv boto3 plotly streamlit-echarts
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn torch fastapi uvicorn streamlit requests confluent-kafka google-generativeai python-dotenv boto3 plotly streamlit-echarts
```

### Docker environment for fullstack demo

```powershell
cd antigravity_system
Copy-Item .env.example .env
docker compose build
```

## How to Run

### A. Chạy EDA

Mục đích: phân tích dữ liệu và sinh báo cáo/biểu đồ.

Windows:

```powershell
$env:DATA_DIR = ".\data"
py -3 .\data\EDA.py
```

macOS/Linux:

```bash
DATA_DIR=./data python3 ./data/EDA.py
```

Kết quả sẽ được sinh vào thư mục `eda_outputs_df26_deep/`.

### B. Chạy training và tạo submission

Mục đích: train model trên `train + val`, suy luận trên `test`, lưu submission và checkpoint.

Windows:

```powershell
$env:DATA_DIR = ".\data"
$env:DETERMINISTIC = "1"
py -3 .\data\Model.py
```

macOS/Linux:

```bash
DATA_DIR=./data DETERMINISTIC=1 python3 ./data/Model.py
```

Các file đầu ra chính:

- `submission_v46_snapshot_regime_dualhead.csv`
- `seqmodel_v46_snapshot_regime_dualhead.ckpt`

### C. Chạy hệ thống dashboard/API

Mục đích: demo what-if simulation và AI agent.

```powershell
cd antigravity_system
Copy-Item .env.example .env
docker compose up --build -d
```

Sau khi chạy:

- Dashboard: `http://localhost:8501`
- API health: `http://localhost:8000/api/health`

Chi tiết hơn xem thêm [antigravity_system/README.md](/C:/Users/HUNG/Downloads/data-20260312T234957Z-1-001/antigravity_system/README.md).

## Data and Resources

- Dữ liệu chính nằm trong thư mục `data/`.
- Model weight demo cho backend nằm trong:
  - `data/v47_snapshot_ep72.pt`
  - `antigravity_system/backend/data/v47_snapshot_ep72.pt`
- Không sử dụng absolute path tới máy cá nhân.
- Nếu cần đổi vị trí dữ liệu, dùng biến môi trường `DATA_DIR`.
- Nếu cần đổi model cho backend, dùng biến môi trường `MODEL_PATH` hoặc `MODEL_FILENAME` trong `.env`.

## Reproducibility Notes

- Script `Model.py` dùng `SEED = 42`.
- Chế độ deterministic có thể bật bằng `DETERMINISTIC=1`.
- `frontend` của hệ thống demo dùng `DEMO_SEED=42` để giảm khác biệt giữa các lần chạy.
- `docker-compose.yml` đã loại bỏ hard-code API key và chuyển sang `.env`.
- `backend` tự retry Kafka nếu broker chưa sẵn sàng ngay lúc khởi động.

## Expected Runtime

- `EDA.py`: vài phút tùy máy.
- `Model.py` trên CPU: có thể khá lâu.
- `docker compose up --build -d`: khoảng `30-60 giây` để hệ thống lên hoàn chỉnh.

## Submission Checklist

- Có `README.md` ở thư mục gốc.
- Có hướng dẫn cài đặt và cách chạy theo thứ tự.
- Không hard-code absolute path.
- Không hard-code secret trong source code.
- Fullstack demo có `.env.example` và `docker-compose.yml`.

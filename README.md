# Real-Time Recommendation Engine

Hệ thống mô phỏng một pipeline gợi ý thời gian thực gồm API phục vụ, xử lý streaming, DAG Airflow, theo dõi MLflow và giám sát Prometheus/Grafana. Tài liệu này hướng dẫn thiết lập và chạy dự án trên máy local (Windows, macOS, Linux).

---

## 1. Yêu cầu hệ thống

- **Python** 3.9+
- **pip** và (khuyến nghị) **virtualenv** hoặc `venv`
- **Docker Desktop** (kèm Docker Compose v2)  
  - Windows: bật WSL2 backend và chia sẻ ổ chứa code (ví dụ ổ `D:`)
- **Make** (tùy chọn, dùng để chạy các lệnh trong `Makefile`. Trên Windows có thể cài `chocolatey install make` hoặc dùng Git Bash)

---

## 2. Chuẩn bị môi trường Python

```powershell
git clone <repo-url>
cd Real-Time-Recommendation-Engine

python -m venv .venv          # hoặc py -3.9 -m venv .venv trên Windows
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate   # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt   # tương đương make install
python scripts/setup.py           # sinh config, seed dữ liệu, tạo topic Kafka
```

> Nếu không muốn cài `make`, có thể chạy trực tiếp các lệnh trong `Makefile`.

---

## 3. Khởi động hạ tầng bằng Docker Compose

```powershell
docker-compose up -d
# hoặc: make start-infra
```

Các dịch vụ chính:

- Kafka (`localhost:9092`)
- Postgres ứng dụng (`localhost:5432`)
- Redis (`localhost:6379`)
- MLflow (`http://localhost:5000`)
- Prometheus (`http://localhost:9090`)
- Grafana (`http://localhost:3000`, tài khoản `admin/admin`)
- Airflow Webserver (`http://localhost:8081`, `admin/admin`)

Dịch vụ Airflow init chỉ chạy một lần để migrate DB và tạo user. Lần đầu khởi động có thể mất ~1–2 phút cho đến khi tất cả healthchecks chuyển trạng thái healthy.

Kiểm tra nhanh:

```powershell
docker ps
docker-compose logs -f   # để theo dõi khi có lỗi
```

---

## 4. Chạy API và xử lý streaming

```powershell
python src/api/recommendation_api.py       # API tại http://localhost:8000 (Swagger /docs)
python src/streaming/feature_processor.py  # Bộ xử lý sự kiện streaming
```

Hoặc dùng lệnh gộp:

```powershell
make start        # start-infra + API
make quick-start  # install -> start-infra -> start-api
```

Khi cần dừng:

```powershell
make stop
# hoặc: docker-compose down && pkill -f recommendation_api.py
```

---

## 5. Demo và huấn luyện

- Demo end-to-end: `python run_demo.py` (hoặc `make demo`) — script này sinh dữ liệu tương tác mẫu, gửi qua streaming và gọi API để kiểm tra kết quả gợi ý.
- Huấn luyện mô hình: `python src/models/train_models.py` (tương đương `make train`).
- Kiểm thử:

```powershell
pytest tests -v --maxfail=1   # hoặc make test
```

---

## 6. Dịch vụ giám sát & công cụ hỗ trợ

| Thành phần  | URL                        | Ghi chú                                |
|-------------|---------------------------|----------------------------------------|
| API         | `http://localhost:8000`   | Swagger tại `/docs`, health `/health` |
| MLflow      | `http://localhost:5000`   | Backend Postgres, artifact local       |
| Grafana     | `http://localhost:3000`   | Dashboard mẫu trong `monitoring/`      |
| Prometheus  | `http://localhost:9090`   | Cấu hình `monitoring/prometheus.yml`   |
| Airflow UI  | `http://localhost:8081`   | Đăng nhập `admin/admin`                |

Các lệnh hữu ích khác (từ `Makefile`):

- `make lint` – flake8 + pylint
- `make format` – black + isort
- `make metrics` – gọi `/metrics` API
- `make logs` – tail log Docker

---

## 7. Gỡ lỗi thường gặp

- **Port bị chiếm dụng**: chỉnh cổng tương ứng trong `docker-compose.yml` hoặc dừng tiến trình đang dùng port.
- **Airflow chưa tạo user**: chạy lại `docker-compose run airflow-init`.
- **Kafka không sẵn sàng**: `docker-compose restart kafka` rồi chạy lại producer.
- **Quyền truy cập volume trên Windows**: mở Docker Desktop → Settings → Resources → File Sharing, thêm ổ chứa repo (ví dụ `D:\`).

---

## 8. Lộ trình đề xuất

1. `make quick-start`
2. Kiểm tra API `http://localhost:8000/docs`
3. Mở Grafana/MLflow để xác nhận dữ liệu
4. Chạy `make demo` để hoàn thiện vòng lặp

Nếu cần hướng dẫn triển khai k8s, CI/CD hoặc Airflow nâng cao, hãy xem thêm các thư mục `k8s/`, `AIRFLOW_SETUP.md`, hoặc yêu cầu thêm tại đây.

Chúc bạn chạy dự án thuận lợi!



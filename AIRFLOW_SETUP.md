# Airflow Setup Guide

Hướng dẫn thiết lập và sử dụng Apache Airflow trong dự án Real-Time Recommendation Engine.

## Yêu cầu

- Docker và Docker Compose đã được cài đặt
- Ít nhất 4GB RAM
- Ít nhất 2 CPU cores
- Ít nhất 10GB dung lượng đĩa

## Khởi động Airflow

### 1. Khởi động tất cả services (bao gồm Airflow)

```bash
docker-compose up -d
```

### 2. Chỉ khởi động Airflow services

```bash
docker-compose up -d postgres-airflow airflow-init airflow-webserver airflow-scheduler
```

### 3. Kiểm tra trạng thái

```bash
docker-compose ps
```

## Truy cập Airflow UI

1. Mở trình duyệt và truy cập: http://localhost:8081
2. Đăng nhập với thông tin:
   - Username: `airflow`
   - Password: `airflow`

## Cấu trúc DAGs

Dự án bao gồm các DAGs sau:

### 1. model_training_pipeline
- **Lịch chạy**: Hàng ngày lúc 2:00 AM
- **Mô tả**: Huấn luyện models SVD và NMF
- **Tasks**:
  - Kiểm tra chất lượng dữ liệu
  - Huấn luyện models
  - Validate models
  - Đăng ký models vào MLflow

### 2. data_processing_pipeline
- **Lịch chạy**: Mỗi 30 phút
- **Mô tả**: Xử lý dữ liệu tương tác người dùng
- **Tasks**:
  - Xử lý interactions từ Kafka
  - Cập nhật feature store
  - Validate dữ liệu

### 3. model_retraining
- **Lịch chạy**: Chủ nhật hàng tuần lúc 3:00 AM
- **Mô tả**: Retrain models khi cần thiết
- **Tasks**:
  - Kiểm tra data drift
  - Kiểm tra hiệu suất model
  - Trigger retraining nếu cần

### 4. system_monitoring
- **Lịch chạy**: Mỗi 15 phút
- **Mô tả**: Giám sát sức khỏe hệ thống
- **Tasks**:
  - Kiểm tra API health
  - Kiểm tra Kafka health
  - Kiểm tra MLflow health
  - Kiểm tra chất lượng dữ liệu

## Quản lý DAGs

### Bật/Tắt DAG

1. Vào Airflow UI
2. Tìm DAG cần quản lý
3. Toggle switch để bật/tắt

### Xem logs

1. Click vào DAG
2. Click vào task
3. Click "Log" để xem logs

### Trigger DAG thủ công

1. Vào DAG page
2. Click nút "Play" (▶️)
3. Chọn "Trigger DAG"

## Tạo DAG mới

1. Tạo file Python mới trong thư mục `dags/`
2. Định nghĩa DAG với các tasks
3. DAG sẽ tự động xuất hiện trong Airflow UI sau vài giây

Ví dụ:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-engineering',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_custom_dag',
    default_args=default_args,
    description='My custom DAG',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

def my_task():
    print("Running my custom task")
    return True

task = PythonOperator(
    task_id='my_task',
    python_callable=my_task,
    dag=dag,
)
```

## Troubleshooting

### DAG không xuất hiện

1. Kiểm tra logs của scheduler:
```bash
docker-compose logs airflow-scheduler
```

2. Kiểm tra syntax của DAG file:
```bash
python dags/your_dag.py
```

### Tasks bị fail

1. Xem logs chi tiết trong Airflow UI
2. Kiểm tra dependencies (Python packages)
3. Kiểm tra kết nối đến services (Kafka, MLflow, etc.)

### Airflow không khởi động

1. Kiểm tra logs:
```bash
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler
```

2. Kiểm tra database connection:
```bash
docker-compose exec postgres-airflow psql -U airflow -d airflow
```

## Cấu hình nâng cao

### Thay đổi executor

Trong `docker-compose.yml`, thay đổi:
```yaml
AIRFLOW__CORE__EXECUTOR=LocalExecutor
```

Các options:
- `LocalExecutor`: Chạy tasks trong cùng process
- `CeleryExecutor`: Chạy tasks trên nhiều workers (cần Redis/RabbitMQ)
- `KubernetesExecutor`: Chạy tasks trên Kubernetes pods

### Thay đổi schedule

Trong DAG file, thay đổi `schedule_interval`:
```python
schedule_interval='0 2 * * *'  # Cron format
schedule_interval='@daily'      # Preset
schedule_interval=timedelta(hours=1)  # Interval
```

## Tài liệu tham khảo

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Airflow DAGs Guide](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dags.html)


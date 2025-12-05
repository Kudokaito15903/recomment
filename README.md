# Real-Time Recommendation Engine

Hệ thống gợi ý real-time với kiến trúc end-to-end từ event ingestion đến model serving, tích hợp MLflow cho model management.

## Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    Event Ingestion Layer                        │
│  (Kafka/Pulsar) - impression, click, purchase, view, session   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Streaming Processor (Enrichment)                    │
│  - Event enrichment với item metadata                            │
│  - Real-time feature engineering                                │
│  - Aggregation và windowing                                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Feature Store (Online)                              │
│  Redis/DynamoDB/Aerospike - Real-time features cho serving      │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│  Serving Layer   │          │  Feedback Loop    │
│  (FastAPI)       │◄─────────│  (Event Logging)  │
└────────┬─────────┘          └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Recommendation Pipeline                             │
│  1. Cache Check (Redis)                                         │
│  2. Candidate Retrieval (FAISS/HNSWlib) - Two-tower             │
│  3. Feature Fetch (Feature Store)                                │
│  4. Ranking (LightGBM/XGBoost/NN)                               │
│  5. Cache Results                                                │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Offline Training Pipeline                            │
│  ETL → Model Training → Evaluation → MLflow Registry            │
└─────────────────────────────────────────────────────────────────┘
```

## Các thành phần chính

### 1. Event Ingestion
- **Kafka/Pulsar**: Nhận tất cả events (impression, click, purchase, view, session)
- **Topics**: `user-events`, `recommendation-feedback`

### 2. Streaming Processor
- **Enrichment**: Thêm metadata, context vào events
- **Real-time Feature Engineering**: Tính toán features theo thời gian thực
- **Aggregation**: Window-based aggregation cho batch processing

### 3. Feature Store (Online)
- **Redis**: Lưu user features real-time cho serving
- **Features**: Base features, enhanced features, metadata
- **TTL Support**: Tự động expire features cũ

### 4. Candidate Retrieval
- **Two-tower Model**: PyTorch-based user/item embeddings
- **FAISS Index**: Approximate Nearest Neighbor search
- **Fallback**: Popular items nếu không có user features

### 5. Ranking Model
- **LightGBM**: Reranker trên user-item feature pairs
- **Low-latency**: Optimized cho serving real-time

### 6. Serving Layer / API Gateway
- **FastAPI**: Stateless microservice
- **Cache**: Redis cache cho recommendations (per-user, per-query)
- **Metrics**: Prometheus metrics endpoint

### 7. Cache Layer
- **Redis**: Cache recommendations với TTL
- **Cache Key**: User ID + num_results + context hash
- **Invalidation**: User-based cache invalidation

### 8. Feedback Loop
- **Event Logging**: Log impressions, interactions về Kafka
- **Online Learning**: Feed events vào stream cho online learning
- **Analytics**: Track recommendation performance

### 9. Offline Training Pipeline
- **ETL**: Extract, Transform, Load từ events
- **Model Training**: Retrieval và Ranking models
- **Evaluation**: Metrics và model comparison
- **MLflow Registry**: Model versioning và deployment

## Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd Real-Time-Recommendation-Engine

# Cài đặt dependencies
pip install -r requirements.txt

# Chuẩn bị dữ liệu demo
python scripts/generate_sample_data.py
```

## Xử lý dữ liệu thực tế

Nếu bạn có dữ liệu thực tế (sản phẩm, người dùng, events), sử dụng script xử lý:

```bash
# Xử lý từ CSV files
python scripts/process_real_data.py \
  --items-file data/raw/products.csv \
  --users-file data/raw/users.csv \
  --events-file data/raw/events.csv \
  --item-id-col product_id \
  --user-id-col customer_id \
  --event-type-col action \
  --timestamp-col created_at \
  --item-metadata-cols category price brand \
  --user-metadata-cols age gender country \
  --output-dir data
```

**Xem hướng dẫn chi tiết**: [docs/DATA_PROCESSING_GUIDE.md](docs/DATA_PROCESSING_GUIDE.md)

Script hỗ trợ:
- ✅ Đọc từ CSV, JSON, Parquet
- ✅ Tự động tạo features nếu chưa có
- ✅ Validate và transform dữ liệu
- ✅ Export sang format chuẩn

## Chạy hệ thống

### 1. Khởi động infrastructure (Docker Compose)

```bash
cd docker
docker compose up -d
```

Dịch vụ:
- `app`: FastAPI (port 8000)
- `redis`: Feature store + Cache (port 6379)
- `kafka` + `zookeeper`: Event backbone
- `mlflow`: MLflow tracking server (port 5000)

### 2. Training models

```bash
# Training retrieval model với MLflow
python scripts/train_retrieval.py

# Training ranking model với MLflow
python scripts/train_ranking.py

# Hoặc chạy offline training pipeline đầy đủ
python scripts/offline_training_pipeline.py
```

### 3. Khởi động streaming processor

```bash
# Kafka realtime
python -m app.streaming --topic user-events --bootstrap localhost:9092

# Hoặc replay từ file
python -m app.streaming --from-file data/sample_events.json
```

### 4. Khởi động API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Recommendations

```bash
# Get recommendations
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_1",
    "num_results": 5,
    "use_cache": true,
    "context": {"device": "mobile"}
  }'

# Response
{
  "user_id": "u_1",
  "latency_ms": 12.5,
  "cached": false,
  "request_id": "abc-123",
  "items": [
    {
      "item_id": "i_1",
      "score": 0.95,
      "metadata": {...}
    }
  ]
}
```

### Log Interactions (Feedback Loop)

```bash
# Log user interaction
curl -X POST http://localhost:8000/interactions?user_id=u_1 \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "i_1",
    "interaction_type": "click",
    "position": 0,
    "request_id": "abc-123"
  }'
```

### User Features

```bash
# Upsert user features
curl -X POST http://localhost:8000/users/u_1/features \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, ...]
  }'
```

### Health & Metrics

```bash
# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

## MLflow Tracking

### Cấu hình

```bash
# Sử dụng MLflow tracking server
export MLFLOW_TRACKING_URI=http://localhost:5000

# Hoặc local file store (mặc định)
export MLFLOW_TRACKING_URI=file:./mlruns
```

### Xem experiments

```bash
# Khởi động MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Truy cập http://localhost:5000
```

### Model Registry

```python
import mlflow

# Load model từ MLflow
model = mlflow.pytorch.load_model("runs:/<run_id>/user_tower")
```

## Kiến trúc chi tiết

### Event Flow

1. **Event Ingestion**: Events được đẩy vào Kafka topics
2. **Streaming Processing**: Events được enrich và transform thành features
3. **Feature Store Update**: Features được cập nhật vào Redis
4. **Recommendation Request**: User gọi API
5. **Cache Check**: Kiểm tra cache trước
6. **Retrieval**: FAISS search để tìm candidates
7. **Ranking**: LightGBM rerank candidates
8. **Response**: Trả về top-k items
9. **Feedback**: Log impression và interactions về Kafka

### Feature Engineering

- **Base Features**: User/item embeddings từ two-tower model
- **Real-time Features**: Click rate, purchase rate, time since last interaction
- **Contextual Features**: Hour of day, day of week, device type
- **Aggregated Features**: Category preferences, item popularity

### Caching Strategy

- **Cache Key**: `user_id:num_results:context_hash`
- **TTL**: 5 minutes (có thể config)
- **Invalidation**: User-based hoặc manual clear
- **Fallback**: In-memory cache nếu Redis unavailable

## Scripts hữu ích

- `scripts/generate_sample_data.py`: Tạo dữ liệu demo
- `scripts/train_retrieval.py`: Training two-tower retrieval model
- `scripts/train_ranking.py`: Training LightGBM ranking model
- `scripts/offline_training_pipeline.py`: ETL + Training + Evaluation pipeline
- `scripts/push_events.py`: Gửi events giả lên Kafka

## Monitoring & Observability

- **Prometheus Metrics**: `/metrics` endpoint
- **MLflow Tracking**: Experiments và model versions
- **Logging**: Structured logging với loguru
- **Health Checks**: `/health` endpoint

## Performance

- **Latency**: < 50ms cho recommendations (với cache)
- **Throughput**: Hỗ trợ hàng nghìn requests/second
- **Cache Hit Rate**: > 80% với TTL hợp lý
- **Feature Store**: Sub-millisecond lookup

## Development

```bash
# Run tests (nếu có)
pytest

# Linting
flake8 app/ scripts/

# Type checking
mypy app/
```

## Production Deployment

1. **Infrastructure**: Kubernetes/Docker Swarm
2. **Scaling**: Horizontal scaling cho API servers
3. **Monitoring**: Prometheus + Grafana
4. **Logging**: ELK stack hoặc CloudWatch
5. **Model Updates**: MLflow model registry + CI/CD

## License

MIT License

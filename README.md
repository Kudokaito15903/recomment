# Realtime AI Recommender

Hệ thống gợi ý gồm 2 tầng (retrieval + ranking) với cập nhật online từ Kafka/Redis và API FastAPI theo thời gian thực.

## Kiến trúc

- **Feature Store**: Redis lưu vector người dùng, fallback in-memory để dev local.
- **Retrieval**: Mô hình two-tower (PyTorch) + FAISS index để tìm ứng viên nhanh.
- **Ranking**: LightGBM reranker trên user/item feature ghép đôi.
- **Streaming**: Consumer Kafka (hoặc phát lại file JSON) cập nhật vector người dùng liên tục.
- **API**: FastAPI + Prometheus metrics (`/metrics`) + endpoint `/recommendations`.

```
Events -> Kafka -> Streaming Updater -> Feature Store
                                       |
User -> FastAPI -> Retrieval (FAISS) -> Ranking (LightGBM) -> Top-N items
```

## Chuẩn bị dữ liệu demo

```bash
python scripts/generate_sample_data.py
python scripts/train_retrieval.py
python scripts/train_ranking.py
```

Các model tạo ở thư mục `models/`.

## Chạy cục bộ

```bash
uvicorn app.main:app --reload
```

Gọi thử:

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_1", "num_results": 5}'
```

## Streaming updater

### Kafka realtime

```bash
python -m app.streaming --topic user-events --bootstrap localhost:9092
```

### Replay file

```bash
python -m app.streaming --from-file data/sample_events.json
```

## Docker Compose

```bash
cd docker
docker compose up --build
```

Dịch vụ chính:

- `app`: FastAPI (port 8000)
- `redis`: feature store
- `kafka` + `zookeeper`: event backbone

## Scripts hữu ích

- `scripts/push_events.py`: gửi event giả lên Kafka.
- `scripts/train_retrieval.py`: huấn luyện two-tower + lưu checkpoint.
- `scripts/train_ranking.py`: huấn luyện LightGBM reranker.

## API chính

- `GET /health`
- `POST /recommendations` `{user_id, num_results}`
- `POST /users/{user_id}/features` – upsert vector tùy chỉnh.
- `GET /metrics` – Prometheus.


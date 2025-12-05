# Hướng dẫn xử lý dữ liệu thực tế

Hướng dẫn chi tiết cách xử lý dữ liệu sản phẩm, người dùng, và events từ dữ liệu thực tế.

## Tổng quan

Hệ thống cần 3 loại dữ liệu chính:
1. **Items (Sản phẩm)**: Thông tin sản phẩm với features
2. **Users (Người dùng)**: Thông tin người dùng với features
3. **Events**: Tương tác giữa user và item (view, click, purchase, etc.)

## Format dữ liệu chuẩn

### 1. Items (sample_items.json)

```json
[
  {
    "item_id": "p_1",
    "features": [0.1, 0.2, 0.3, ...],  // Vector features (16-32 dimensions)
    "category": "electronics",
    "price": 99.99,
    "popularity": 100
  }
]
```

**Yêu cầu:**
- `item_id`: ID duy nhất của sản phẩm (bắt buộc)
- `features`: Vector features (bắt buộc, nếu không có sẽ tự động tạo)
- `category`: Danh mục sản phẩm (khuyến nghị)
- `price`: Giá sản phẩm (khuyến nghị)
- `popularity`: Độ phổ biến (khuyến nghị)

### 2. Users (sample_users.json)

```json
[
  {
    "user_id": "u_1",
    "features": [0.1, 0.2, 0.3, ...],  // Vector features (16-32 dimensions)
    "age": 25,
    "country": "VN"
  }
]
```

**Yêu cầu:**
- `user_id`: ID duy nhất của người dùng (bắt buộc)
- `features`: Vector features (bắt buộc, nếu không có sẽ tự động tạo)
- Các metadata khác: tùy chọn

### 3. Events (sample_events.json)

```json
[
  {
    "user_id": "u_1",
    "item_id": "p_1",
    "event_type": "click",  // view, click, purchase, impression
    "timestamp": 1700000000000  // milliseconds
  }
]
```

**Yêu cầu:**
- `user_id`: ID người dùng (bắt buộc)
- `item_id`: ID sản phẩm (bắt buộc)
- `event_type`: Loại event (bắt buộc)
- `timestamp`: Thời gian (khuyến nghị, nếu không có sẽ tự động tạo)

## Xử lý dữ liệu từ nhiều nguồn

### Cách 1: Sử dụng script tự động

Script `scripts/process_real_data.py` hỗ trợ đọc từ CSV, JSON, Parquet:

```bash
# Ví dụ với CSV
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

### Cách 2: Xử lý thủ công với Python

```python
import pandas as pd
import json
import numpy as np
from scripts.process_real_data import DataProcessor

# Load dữ liệu
items_df = pd.read_csv("products.csv")
users_df = pd.read_csv("users.csv")
events_df = pd.read_csv("events.csv")

# Khởi tạo processor
processor = DataProcessor(output_dir="data")

# Xử lý items
items = processor.process_items(
    items_df,
    item_id_col="product_id",
    feature_cols=["price", "rating", "sales_count"],  # Nếu có sẵn
    metadata_cols=["category", "brand", "price"],
    auto_generate_features=True,  # Tự động tạo nếu không có
    feature_dim=16
)

# Xử lý users
users = processor.process_users(
    users_df,
    user_id_col="customer_id",
    feature_cols=["age", "total_purchases"],  # Nếu có sẵn
    metadata_cols=["age", "country", "gender"],
    auto_generate_features=True,
    feature_dim=16
)

# Xử lý events
events = processor.process_events(
    events_df,
    user_id_col="customer_id",
    item_id_col="product_id",
    event_type_col="action",
    timestamp_col="created_at"
)

# Lưu kết quả
processor.save_processed_data(items, users, events)
```

### Cách 3: Từ Database

```python
import pandas as pd
from sqlalchemy import create_engine
from scripts.process_real_data import DataProcessor

# Kết nối database
engine = create_engine("postgresql://user:pass@localhost/dbname")

# Load dữ liệu
items_df = pd.read_sql("SELECT * FROM products", engine)
users_df = pd.read_sql("SELECT * FROM users", engine)
events_df = pd.read_sql("SELECT * FROM events", engine)

# Xử lý như trên
processor = DataProcessor()
items = processor.process_items(items_df, ...)
users = processor.process_users(users_df, ...)
events = processor.process_events(events_df, ...)
processor.save_processed_data(items, users, events)
```

## Tạo Features tự động

Nếu dữ liệu của bạn chưa có features (vector embeddings), hệ thống sẽ tự động tạo từ metadata:

### Items Features
- Từ các cột numeric: price, rating, sales_count, etc.
- Encode category thành số
- Normalize về [0, 1]

### Users Features
- Từ các cột numeric: age, total_purchases, etc.
- Encode country/region thành số
- Normalize về [0, 1]

### Sử dụng Pre-trained Embeddings (Khuyến nghị)

Để có features tốt hơn, bạn có thể sử dụng pre-trained models:

```python
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Tạo embeddings cho items từ description
items_df = pd.read_csv("products.csv")
descriptions = items_df["description"].fillna("").tolist()
item_embeddings = model.encode(descriptions)

# Gán vào items
for i, item in enumerate(items):
    item["features"] = item_embeddings[i].tolist()
```

## Ví dụ cụ thể

### Ví dụ 1: E-commerce data

```bash
# Dữ liệu có cấu trúc:
# products.csv: product_id, name, category, price, description
# customers.csv: customer_id, age, gender, country
# interactions.csv: customer_id, product_id, action, timestamp

python scripts/process_real_data.py \
  --items-file products.csv \
  --users-file customers.csv \
  --events-file interactions.csv \
  --item-id-col product_id \
  --user-id-col customer_id \
  --event-type-col action \
  --timestamp-col timestamp \
  --item-metadata-cols category price \
  --user-metadata-cols age gender country
```

### Ví dụ 2: Chỉ có events, không có users

```python
# Tạo users từ events
events_df = pd.read_csv("events.csv")
unique_users = events_df["user_id"].unique()

# Tạo users với random features
users = []
for user_id in unique_users:
    users.append({
        "user_id": str(user_id),
        "features": np.random.rand(16).tolist()  # Sẽ được cập nhật sau khi training
    })

# Lưu users
with open("data/sample_users.json", "w") as f:
    json.dump(users, f, indent=2)
```

### Ví dụ 3: Dữ liệu từ API

```python
import requests
import pandas as pd

# Fetch từ API
response = requests.get("https://api.example.com/products")
items_data = response.json()

# Convert sang DataFrame
items_df = pd.DataFrame(items_data)

# Xử lý
processor = DataProcessor()
items = processor.process_items(items_df, item_id_col="id")
```

## Validation và Quality Check

Sau khi xử lý, kiểm tra chất lượng dữ liệu:

```python
import json

# Load dữ liệu đã xử lý
with open("data/sample_items.json") as f:
    items = json.load(f)

# Kiểm tra
print(f"Total items: {len(items)}")
print(f"Items with features: {sum(1 for i in items if 'features' in i)}")
print(f"Items with category: {sum(1 for i in items if 'category' in i)}")

# Kiểm tra feature dimensions
feature_dims = [len(i['features']) for i in items if 'features' in i]
print(f"Feature dimensions: {set(feature_dims)}")
```

## Best Practices

1. **Features Consistency**: Đảm bảo tất cả items/users có cùng số chiều features
2. **ID Format**: Sử dụng string cho IDs, tránh special characters
3. **Event Types**: Chuẩn hóa event types (lowercase, consistent naming)
4. **Timestamps**: Sử dụng milliseconds (Unix timestamp * 1000)
5. **Missing Data**: Xử lý missing values trước khi xử lý
6. **Data Quality**: Validate dữ liệu trước khi training

## Troubleshooting

### Lỗi: "Column not found"
- Kiểm tra tên cột với `--item-id-col`, `--user-id-col`
- Xem cấu trúc file với `pd.read_csv("file.csv").columns`

### Lỗi: "Features dimension mismatch"
- Đảm bảo tất cả items/users có cùng số chiều
- Sử dụng `--feature-dim` để set số chiều

### Lỗi: "Invalid timestamp"
- Kiểm tra format timestamp
- Script sẽ tự động tạo timestamp nếu không có

## Next Steps

Sau khi xử lý xong dữ liệu:

1. **Training models**:
   ```bash
   python scripts/train_retrieval.py
   python scripts/train_ranking.py
   ```

2. **Load vào feature store**:
   ```python
   from app.feature_store import FeatureStore
   store = FeatureStore()
   store.bulk_load_from_file("data/sample_users.json")
   ```

3. **Test recommendations**:
   ```bash
   curl -X POST http://localhost:8000/recommendations \
     -H "Content-Type: application/json" \
     -d '{"user_id": "u_1", "num_results": 5}'
   ```


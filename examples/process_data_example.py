"""
Ví dụ cụ thể về cách xử lý dữ liệu thực tế.

Giả sử bạn có dữ liệu từ database hoặc CSV files.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.process_real_data import DataProcessor


def example_1_from_csv():
    """Ví dụ 1: Xử lý dữ liệu từ CSV files."""
    print("=" * 60)
    print("VÍ DỤ 1: Xử lý dữ liệu từ CSV")
    print("=" * 60)

    # Giả sử bạn có các file CSV:
    # - products.csv: product_id, name, category, price, description
    # - customers.csv: customer_id, age, gender, country, total_purchases
    # - interactions.csv: customer_id, product_id, action, timestamp

    processor = DataProcessor(output_dir="data")

    # Load và xử lý items
    print("\n📦 Xử lý sản phẩm...")
    items_df = processor.load_data("data/raw/products.csv", file_type="csv")
    items = processor.process_items(
        items_df,
        item_id_col="product_id",
        feature_cols=None,  # Tự động tạo từ metadata
        metadata_cols=["category", "price", "brand"],
        auto_generate_features=True,
        feature_dim=16,
    )

    # Load và xử lý users
    print("\n👥 Xử lý người dùng...")
    users_df = processor.load_data("data/raw/customers.csv", file_type="csv")
    users = processor.process_users(
        users_df,
        user_id_col="customer_id",
        feature_cols=["age", "total_purchases"],  # Sử dụng các cột này làm features
        metadata_cols=["age", "country", "gender"],
        auto_generate_features=True,
        feature_dim=16,
    )

    # Load và xử lý events
    print("\n📊 Xử lý events...")
    events_df = processor.load_data("data/raw/interactions.csv", file_type="csv")
    events = processor.process_events(
        events_df,
        user_id_col="customer_id",
        item_id_col="product_id",
        event_type_col="action",
        timestamp_col="timestamp",
    )

    # Lưu kết quả
    processor.save_processed_data(items, users, events)
    print("\n✅ Hoàn tất!")


def example_2_from_database():
    """Ví dụ 2: Xử lý dữ liệu từ database."""
    print("=" * 60)
    print("VÍ DỤ 2: Xử lý dữ liệu từ Database")
    print("=" * 60)

    try:
        from sqlalchemy import create_engine

        # Kết nối database
        engine = create_engine("postgresql://user:pass@localhost/dbname")

        processor = DataProcessor(output_dir="data")

        # Load từ database
        print("\n📦 Load sản phẩm từ database...")
        items_df = pd.read_sql("SELECT * FROM products", engine)

        print("\n👥 Load người dùng từ database...")
        users_df = pd.read_sql("SELECT * FROM users", engine)

        print("\n📊 Load events từ database...")
        events_df = pd.read_sql(
            """
            SELECT customer_id as user_id, 
                   product_id as item_id,
                   action as event_type,
                   created_at as timestamp
            FROM interactions
            """,
            engine,
        )

        # Xử lý
        items = processor.process_items(items_df, item_id_col="product_id")
        users = processor.process_users(users_df, user_id_col="customer_id")
        events = processor.process_events(
            events_df,
            user_id_col="user_id",
            item_id_col="item_id",
            event_type_col="event_type",
            timestamp_col="timestamp",
        )

        # Lưu
        processor.save_processed_data(items, users, events)
        print("\n✅ Hoàn tất!")

    except ImportError:
        print("⚠️  Cần cài đặt: pip install sqlalchemy psycopg2")


def example_3_manual_processing():
    """Ví dụ 3: Xử lý thủ công với dữ liệu tùy chỉnh."""
    print("=" * 60)
    print("VÍ DỤ 3: Xử lý thủ công")
    print("=" * 60)

    # Giả sử bạn có dữ liệu từ API hoặc nguồn khác
    items_data = [
        {
            "id": "prod_1",
            "name": "Laptop",
            "category": "electronics",
            "price": 999.99,
            "rating": 4.5,
            "sales": 1000,
        },
        {
            "id": "prod_2",
            "name": "Phone",
            "category": "electronics",
            "price": 599.99,
            "rating": 4.8,
            "sales": 2000,
        },
    ]

    users_data = [
        {"id": "user_1", "age": 25, "country": "VN", "total_orders": 10},
        {"id": "user_2", "age": 30, "country": "US", "total_orders": 5},
    ]

    events_data = [
        {"user_id": "user_1", "item_id": "prod_1", "action": "click"},
        {"user_id": "user_1", "item_id": "prod_2", "action": "purchase"},
        {"user_id": "user_2", "item_id": "prod_1", "action": "view"},
    ]

    # Convert sang DataFrame
    items_df = pd.DataFrame(items_data)
    users_df = pd.DataFrame(users_data)
    events_df = pd.DataFrame(events_data)

    processor = DataProcessor(output_dir="data")

    # Xử lý
    items = processor.process_items(
        items_df,
        item_id_col="id",
        feature_cols=["price", "rating", "sales"],  # Sử dụng các cột này
        metadata_cols=["category", "price"],
    )

    users = processor.process_users(
        users_df,
        user_id_col="id",
        feature_cols=["age", "total_orders"],
        metadata_cols=["age", "country"],
    )

    events = processor.process_events(
        events_df,
        user_id_col="user_id",
        item_id_col="item_id",
        event_type_col="action",
    )

    # Lưu
    processor.save_processed_data(items, users, events)
    print("\n✅ Hoàn tất!")

    # In ra để xem kết quả
    print("\n📦 Items sample:")
    print(json.dumps(items[0], indent=2))
    print("\n👥 Users sample:")
    print(json.dumps(users[0], indent=2))
    print("\n📊 Events sample:")
    print(json.dumps(events[0], indent=2))


def example_4_only_events():
    """Ví dụ 4: Chỉ có events, tạo users và items từ events."""
    print("=" * 60)
    print("VÍ DỤ 4: Tạo users/items từ events")
    print("=" * 60)

    # Load events
    with open("data/sample_events.json") as f:
        events = json.load(f)

    # Extract unique users và items
    unique_users = set(e["user_id"] for e in events)
    unique_items = set(e["item_id"] for e in events)

    print(f"Found {len(unique_users)} unique users")
    print(f"Found {len(unique_items)} unique items")

    # Tạo users với random features (sẽ được cập nhật sau khi training)
    users = []
    for user_id in unique_users:
        users.append(
            {
                "user_id": user_id,
                "features": np.random.rand(16).tolist(),  # Random features
            }
        )

    # Tạo items với random features
    items = []
    for item_id in unique_items:
        items.append(
            {
                "item_id": item_id,
                "features": np.random.rand(16).tolist(),  # Random features
                "category": "unknown",
                "price": 0.0,
                "popularity": 0,
            }
        )

    # Lưu
    processor = DataProcessor(output_dir="data")
    processor.save_processed_data(items, users, events)

    print("\n✅ Đã tạo users và items từ events!")
    print("⚠️  Lưu ý: Features là random, cần training model để có features tốt hơn")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ví dụ xử lý dữ liệu")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
        help="Chọn ví dụ (1: CSV, 2: Database, 3: Manual, 4: Only Events)",
    )
    args = parser.parse_args()

    if args.example == 1:
        example_1_from_csv()
    elif args.example == 2:
        example_2_from_database()
    elif args.example == 3:
        example_3_manual_processing()
    elif args.example == 4:
        example_4_only_events()


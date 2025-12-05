"""
Script xử lý dữ liệu thực tế (sản phẩm, người dùng, events) cho hệ thống recommendation.

Hỗ trợ:
- Đọc từ CSV, JSON, Parquet, hoặc database
- Tạo features tự động nếu chưa có
- Validate và transform dữ liệu
- Export sang format chuẩn cho hệ thống
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


class DataProcessor:
    """Xử lý và transform dữ liệu thực tế sang format chuẩn."""

    def __init__(self, output_dir: Path = Path("data")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_items(
        self,
        items_df: pd.DataFrame,
        item_id_col: str = "item_id",
        feature_cols: Optional[List[str]] = None,
        metadata_cols: Optional[List[str]] = None,
        auto_generate_features: bool = True,
        feature_dim: int = 16,
    ) -> List[Dict]:
        """
        Xử lý dữ liệu sản phẩm.

        Args:
            items_df: DataFrame chứa dữ liệu sản phẩm
            item_id_col: Tên cột chứa item ID
            feature_cols: Danh sách cột dùng làm features (nếu None, tự động tạo)
            metadata_cols: Danh sách cột metadata (category, price, etc.)
            auto_generate_features: Tự động tạo features nếu chưa có
            feature_dim: Số chiều features nếu tự động tạo
        """
        logger.info(f"Processing {len(items_df)} items")

        # Validate item_id column
        if item_id_col not in items_df.columns:
            raise ValueError(f"Column '{item_id_col}' not found in items data")

        items = []
        for _, row in items_df.iterrows():
            item = {"item_id": str(row[item_id_col])}

            # Xử lý features
            if feature_cols:
                # Sử dụng các cột được chỉ định
                features = []
                for col in feature_cols:
                    if col in row:
                        val = row[col]
                        # Convert to float
                        try:
                            features.append(float(val))
                        except (ValueError, TypeError):
                            features.append(0.0)
                item["features"] = features
            elif auto_generate_features:
                # Tự động tạo features từ metadata
                features = self._generate_item_features(row, feature_dim, metadata_cols)
                item["features"] = features
            else:
                # Nếu không có features, tạo random (fallback)
                logger.warning(f"No features for item {item['item_id']}, generating random")
                item["features"] = np.random.rand(feature_dim).tolist()

            # Xử lý metadata
            if metadata_cols:
                for col in metadata_cols:
                    if col in row:
                        item[col] = row[col]

            # Đảm bảo có các trường cơ bản
            if "category" not in item:
                item["category"] = "unknown"
            if "price" not in item:
                item["price"] = 0.0
            if "popularity" not in item:
                item["popularity"] = 0

            items.append(item)

        logger.info(f"Processed {len(items)} items")
        return items

    def _generate_item_features(
        self, row: pd.Series, dim: int, metadata_cols: Optional[List[str]]
    ) -> List[float]:
        """Tạo features từ metadata của item."""
        features = []

        # Sử dụng các cột numeric
        numeric_cols = row.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:dim]:
            val = float(row[col]) if pd.notna(row[col]) else 0.0
            # Normalize về [0, 1]
            features.append(val)

        # Nếu chưa đủ dim, thêm từ metadata
        if len(features) < dim:
            # Encode category nếu có
            if "category" in row and pd.notna(row["category"]):
                category_hash = hash(str(row["category"])) % 1000 / 1000.0
                features.append(category_hash)

            # Thêm random để đủ dim
            while len(features) < dim:
                features.append(np.random.rand())

        return features[:dim]

    def process_users(
        self,
        users_df: pd.DataFrame,
        user_id_col: str = "user_id",
        feature_cols: Optional[List[str]] = None,
        metadata_cols: Optional[List[str]] = None,
        auto_generate_features: bool = True,
        feature_dim: int = 16,
    ) -> List[Dict]:
        """
        Xử lý dữ liệu người dùng.

        Args:
            users_df: DataFrame chứa dữ liệu người dùng
            user_id_col: Tên cột chứa user ID
            feature_cols: Danh sách cột dùng làm features
            metadata_cols: Danh sách cột metadata
            auto_generate_features: Tự động tạo features nếu chưa có
            feature_dim: Số chiều features nếu tự động tạo
        """
        logger.info(f"Processing {len(users_df)} users")

        if user_id_col not in users_df.columns:
            raise ValueError(f"Column '{user_id_col}' not found in users data")

        users = []
        for _, row in users_df.iterrows():
            user = {"user_id": str(row[user_id_col])}

            # Xử lý features
            if feature_cols:
                features = []
                for col in feature_cols:
                    if col in row:
                        val = row[col]
                        try:
                            features.append(float(val))
                        except (ValueError, TypeError):
                            features.append(0.0)
                user["features"] = features
            elif auto_generate_features:
                features = self._generate_user_features(row, feature_dim, metadata_cols)
                user["features"] = features
            else:
                logger.warning(f"No features for user {user['user_id']}, generating random")
                user["features"] = np.random.rand(feature_dim).tolist()

            # Xử lý metadata
            if metadata_cols:
                for col in metadata_cols:
                    if col in row:
                        user[col] = row[col]

            users.append(user)

        logger.info(f"Processed {len(users)} users")
        return users

    def _generate_user_features(
        self, row: pd.Series, dim: int, metadata_cols: Optional[List[str]]
    ) -> List[float]:
        """Tạo features từ metadata của user."""
        features = []

        # Sử dụng các cột numeric
        numeric_cols = row.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:dim]:
            val = float(row[col]) if pd.notna(row[col]) else 0.0
            features.append(val)

        # Encode categorical nếu có
        if "country" in row and pd.notna(row["country"]):
            country_hash = hash(str(row["country"])) % 1000 / 1000.0
            features.append(country_hash)

        # Thêm random để đủ dim
        while len(features) < dim:
            features.append(np.random.rand())

        return features[:dim]

    def process_events(
        self,
        events_df: pd.DataFrame,
        user_id_col: str = "user_id",
        item_id_col: str = "item_id",
        event_type_col: str = "event_type",
        timestamp_col: Optional[str] = None,
    ) -> List[Dict]:
        """
        Xử lý dữ liệu events.

        Args:
            events_df: DataFrame chứa events
            user_id_col: Tên cột user ID
            item_id_col: Tên cột item ID
            event_type_col: Tên cột event type
            timestamp_col: Tên cột timestamp (optional)
        """
        logger.info(f"Processing {len(events_df)} events")

        required_cols = [user_id_col, item_id_col, event_type_col]
        missing_cols = [col for col in required_cols if col not in events_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        events = []
        for _, row in events_df.iterrows():
            event = {
                "user_id": str(row[user_id_col]),
                "item_id": str(row[item_id_col]),
                "event_type": str(row[event_type_col]).lower(),
            }

            # Xử lý timestamp
            if timestamp_col and timestamp_col in row:
                ts = row[timestamp_col]
                if pd.notna(ts):
                    # Convert to milliseconds
                    if isinstance(ts, (int, float)):
                        # Nếu đã là timestamp, giả sử là seconds
                        if ts < 1e12:  # Less than year 2001 in ms
                            event["timestamp"] = int(ts * 1000)
                        else:
                            event["timestamp"] = int(ts)
                    else:
                        # Try to parse datetime
                        try:
                            dt = pd.to_datetime(ts)
                            event["timestamp"] = int(dt.timestamp() * 1000)
                        except:
                            event["timestamp"] = int(pd.Timestamp.now().timestamp() * 1000)
            else:
                # Generate timestamp nếu không có
                event["timestamp"] = int(pd.Timestamp.now().timestamp() * 1000)

            events.append(event)

        logger.info(f"Processed {len(events)} events")
        return events

    def load_data(
        self, file_path: Path, file_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Load dữ liệu từ file (CSV, JSON, Parquet)."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect file type
        if file_type is None:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                file_type = "csv"
            elif suffix == ".json":
                file_type = "json"
            elif suffix in [".parquet", ".pq"]:
                file_type = "parquet"
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

        logger.info(f"Loading {file_type} file: {file_path}")

        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "json":
            df = pd.read_json(file_path)
        elif file_type == "parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def save_processed_data(
        self, items: List[Dict], users: List[Dict], events: List[Dict]
    ) -> None:
        """Lưu dữ liệu đã xử lý."""
        # Save items
        items_path = self.output_dir / "sample_items.json"
        with open(items_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(items)} items to {items_path}")

        # Save users
        users_path = self.output_dir / "sample_users.json"
        with open(users_path, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(users)} users to {users_path}")

        # Save events
        events_path = self.output_dir / "sample_events.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(events)} events to {events_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Xử lý dữ liệu thực tế cho hệ thống recommendation"
    )
    parser.add_argument(
        "--items-file",
        type=Path,
        required=True,
        help="File chứa dữ liệu sản phẩm (CSV/JSON/Parquet)",
    )
    parser.add_argument(
        "--users-file",
        type=Path,
        help="File chứa dữ liệu người dùng (CSV/JSON/Parquet)",
    )
    parser.add_argument(
        "--events-file",
        type=Path,
        required=True,
        help="File chứa dữ liệu events (CSV/JSON/Parquet)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="data", help="Thư mục output"
    )

    # Column mapping
    parser.add_argument("--item-id-col", default="item_id", help="Tên cột item ID")
    parser.add_argument("--user-id-col", default="user_id", help="Tên cột user ID")
    parser.add_argument(
        "--event-type-col", default="event_type", help="Tên cột event type"
    )
    parser.add_argument(
        "--timestamp-col", help="Tên cột timestamp (optional)"
    )

    # Features
    parser.add_argument(
        "--item-feature-cols",
        nargs="+",
        help="Danh sách cột dùng làm features cho items",
    )
    parser.add_argument(
        "--user-feature-cols",
        nargs="+",
        help="Danh sách cột dùng làm features cho users",
    )
    parser.add_argument(
        "--item-metadata-cols",
        nargs="+",
        help="Danh sách cột metadata cho items (category, price, etc.)",
    )
    parser.add_argument(
        "--user-metadata-cols",
        nargs="+",
        help="Danh sách cột metadata cho users (age, country, etc.)",
    )
    parser.add_argument(
        "--feature-dim", type=int, default=16, help="Số chiều features nếu tự động tạo"
    )

    args = parser.parse_args()

    processor = DataProcessor(output_dir=args.output_dir)

    # Load và xử lý items
    items_df = processor.load_data(args.items_file)
    items = processor.process_items(
        items_df,
        item_id_col=args.item_id_col,
        feature_cols=args.item_feature_cols,
        metadata_cols=args.item_metadata_cols,
        feature_dim=args.feature_dim,
    )

    # Load và xử lý users (nếu có)
    users = []
    if args.users_file:
        users_df = processor.load_data(args.users_file)
        users = processor.process_users(
            users_df,
            user_id_col=args.user_id_col,
            feature_cols=args.user_feature_cols,
            metadata_cols=args.user_metadata_cols,
            feature_dim=args.feature_dim,
        )

    # Load và xử lý events
    events_df = processor.load_data(args.events_file)
    events = processor.process_events(
        events_df,
        user_id_col=args.user_id_col,
        item_id_col=args.item_id_col,
        event_type_col=args.event_type_col,
        timestamp_col=args.timestamp_col,
    )

    # Lưu dữ liệu đã xử lý
    processor.save_processed_data(items, users, events)

    logger.info("✅ Xử lý dữ liệu hoàn tất!")
    logger.info(f"📦 Items: {len(items)}")
    logger.info(f"👥 Users: {len(users)}")
    logger.info(f"📊 Events: {len(events)}")


if __name__ == "__main__":
    main()


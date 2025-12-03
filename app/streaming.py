import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from kafka import KafkaConsumer
from loguru import logger

from app.feature_store import FeatureStore


class StreamingUpdater:
    """Consumes user events and updates user feature vectors online."""

    def __init__(self, items_path: str = "data/sample_items.json"):
        self.feature_store = FeatureStore()
        with open(items_path) as f:
            items = json.load(f)
        self.item_vectors: Dict[str, np.ndarray] = {
            item["item_id"]: np.array(item["features"], dtype=np.float32) for item in items
        }

    def update_from_event(self, event: Dict) -> None:
        user_id = event["user_id"]
        item_id = event.get("item_id")
        item_vector = self.item_vectors.get(item_id)
        if item_vector is None:
            return

        current = self.feature_store.get_user_features(user_id)
        if current is None:
            updated = item_vector
        else:
            alpha = 0.8 if event.get("event_type") == "view" else 0.5
            updated = alpha * current + (1 - alpha) * item_vector
        self.feature_store.set_user_features(user_id, updated)
        logger.info(f"Updated features for user {user_id} from event {event.get('event_type')}")

    def process_events(self, events: Iterable[Dict]) -> None:
        for event in events:
            self.update_from_event(event)


def kafka_event_stream(topic: str, bootstrap_servers: str) -> Iterable[Dict]:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id="recommender_group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
    )
    logger.info(f"Consuming Kafka topic '{topic}' on {bootstrap_servers}")
    for message in consumer:
        yield message.value


def file_event_stream(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        events = json.load(f)
    random.shuffle(events)
    logger.info(f"Replaying {len(events)} events from {path}")
    for event in events:
        yield event


def main():
    parser = argparse.ArgumentParser(description="Realtime feature updater")
    parser.add_argument("--topic", default="user-events")
    parser.add_argument("--bootstrap", default="http://localhost:9092")
    parser.add_argument(
        "--from-file",
        type=Path,
        help="Replay events from JSON instead of consuming Kafka",
    )
    args = parser.parse_args()

    updater = StreamingUpdater()
    if args.from_file:
        events = file_event_stream(args.from_file)
    else:
        events = kafka_event_stream(args.topic, args.bootstrap)
    updater.process_events(events)


if __name__ == "__main__":
    main()

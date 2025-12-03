import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import redis
from loguru import logger


class FeatureStore:
    """Read/write dense user feature vectors backed by Redis with in-memory fallback."""

    def __init__(self, redis_url: Optional[str] = None, namespace: str = "recommender"):
        self.namespace = namespace
        self._memory_store = {}
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/")
        try:
            self.redis = redis.from_url(redis_url)
            self.redis.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except redis.RedisError as exc:
            logger.warning(f"Redis unavailable ({exc}); falling back to in-memory store")
            self.redis = None

    def _key(self, user_id: str) -> str:
        return f"{self.namespace}:user:{user_id}:features"

    def get_user_features(self, user_id: str) -> Optional[np.ndarray]:
        if self.redis:
            val = self.redis.get(self._key(user_id))
            if val is None:
                return None
            return np.frombuffer(val, dtype=np.float32)
        return self._memory_store.get(user_id)

    def set_user_features(self, user_id: str, features: Iterable[float]) -> None:
        arr = np.array(features, dtype=np.float32)
        if self.redis:
            self.redis.set(self._key(user_id), arr.tobytes())
        self._memory_store[user_id] = arr

    def bulk_load_from_file(self, path: str | Path, overwrite: bool = False) -> int:
        """Seed the store with user vectors from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Feature seed file {path} not found")
            return 0

        loaded = 0
        with path.open() as f:
            payload = json.load(f)
        for record in payload:
            user_id = record.get("user_id")
            features = record.get("features")
            if not user_id or features is None:
                continue
            if not overwrite and self.get_user_features(user_id) is not None:
                continue
            self.set_user_features(user_id, features)
            loaded += 1
        logger.info(f"Seeded {loaded} user feature vectors from {path}")
        return loaded

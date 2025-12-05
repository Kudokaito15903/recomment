import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import redis
from loguru import logger


class FeatureStore:
    """Online feature store for real-time serving with Redis backend and in-memory fallback."""

    def __init__(self, redis_url: Optional[str] = None, namespace: str = "recommender"):
        self.namespace = namespace
        self._memory_store = {}
        self._memory_metadata = {}
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/")
        try:
            self.redis = redis.from_url(redis_url)
            self.redis.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except redis.RedisError as exc:
            logger.warning(f"Redis unavailable ({exc}); falling back to in-memory store")
            self.redis = None

    def _key(self, user_id: str, feature_type: str = "base") -> str:
        """Generate Redis key for user feature."""
        return f"{self.namespace}:user:{user_id}:features:{feature_type}"

    def _metadata_key(self, user_id: str) -> str:
        """Generate Redis key for user metadata."""
        return f"{self.namespace}:user:{user_id}:metadata"

    def get_user_features(
        self, user_id: str, feature_type: str = "base"
    ) -> Optional[np.ndarray]:
        """Get user features by type."""
        if self.redis:
            val = self.redis.get(self._key(user_id, feature_type))
            if val is None:
                return None
            return np.frombuffer(val, dtype=np.float32)
        return self._memory_store.get((user_id, feature_type))

    def set_user_features(
        self,
        user_id: str,
        features: Iterable[float],
        feature_type: str = "base",
        ttl: Optional[int] = None,
    ) -> None:
        """Set user features with optional TTL."""
        arr = np.array(features, dtype=np.float32)
        key = self._key(user_id, feature_type)
        
        if self.redis:
            if ttl:
                self.redis.setex(key, ttl, arr.tobytes())
            else:
                self.redis.set(key, arr.tobytes())
        self._memory_store[(user_id, feature_type)] = arr

    def get_user_metadata(self, user_id: str) -> Optional[Dict]:
        """Get user metadata (stats, preferences, etc.)."""
        if self.redis:
            val = self.redis.get(self._metadata_key(user_id))
            if val is None:
                return None
            return json.loads(val)
        return self._memory_metadata.get(user_id)

    def set_user_metadata(self, user_id: str, metadata: Dict, ttl: Optional[int] = None) -> None:
        """Set user metadata."""
        key = self._metadata_key(user_id)
        
        if self.redis:
            if ttl:
                self.redis.setex(key, ttl, json.dumps(metadata))
            else:
                self.redis.set(key, json.dumps(metadata))
        self._memory_metadata[user_id] = metadata

    def update_user_metadata(self, user_id: str, updates: Dict) -> None:
        """Update user metadata (merge with existing)."""
        current = self.get_user_metadata(user_id) or {}
        current.update(updates)
        self.set_user_metadata(user_id, current)

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

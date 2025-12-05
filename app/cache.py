"""Caching layer for recommendations and features."""
import hashlib
import json
import os
from typing import Dict, List, Optional

import redis
from loguru import logger


class RecommendationCache:
    """Cache for recommendation results to reduce latency."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        namespace: str = "recommender",
        ttl_seconds: int = 300,  # 5 minutes default TTL
    ):
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self._memory_cache = {}
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/")
        try:
            self.redis = redis.from_url(redis_url)
            self.redis.ping()
            logger.info(f"Connected to Redis cache at {redis_url}")
        except redis.RedisError as exc:
            logger.warning(f"Redis cache unavailable ({exc}); falling back to in-memory cache")
            self.redis = None

    def _cache_key(self, user_id: str, num_results: int, context: Optional[Dict] = None) -> str:
        """Generate cache key from user_id, num_results, and optional context."""
        key_parts = [user_id, str(num_results)]
        if context:
            # Include relevant context fields in cache key
            context_str = json.dumps(context, sort_keys=True)
            key_parts.append(context_str)
        key_string = ":".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{self.namespace}:rec:{key_hash}"

    def get(self, user_id: str, num_results: int, context: Optional[Dict] = None) -> Optional[List[Dict]]:
        """Get cached recommendations."""
        cache_key = self._cache_key(user_id, num_results, context)
        
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Error reading from Redis cache: {e}")
        
        # Fallback to memory cache
        return self._memory_cache.get(cache_key)

    def set(
        self,
        user_id: str,
        num_results: int,
        recommendations: List[Dict],
        context: Optional[Dict] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Cache recommendations."""
        cache_key = self._cache_key(user_id, num_results, context)
        ttl = ttl or self.ttl_seconds
        
        if self.redis:
            try:
                self.redis.setex(
                    cache_key,
                    ttl,
                    json.dumps(recommendations),
                )
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        
        # Also store in memory cache
        self._memory_cache[cache_key] = recommendations

    def invalidate_user(self, user_id: str) -> None:
        """Invalidate all cached recommendations for a user."""
        pattern = f"{self.namespace}:rec:*"
        
        if self.redis:
            try:
                # Note: This is a simple implementation. In production, consider using SCAN
                # for better performance with large key sets
                keys = self.redis.keys(pattern)
                if keys:
                    # Check if keys match user_id (would need to store user_id mapping)
                    # For now, we'll just log
                    logger.info(f"Cache invalidation requested for user {user_id}")
            except Exception as e:
                logger.warning(f"Error invalidating cache: {e}")
        
        # Clear memory cache entries (simplified)
        keys_to_remove = [k for k in self._memory_cache.keys() if user_id in k]
        for key in keys_to_remove:
            del self._memory_cache[key]

    def clear(self) -> None:
        """Clear all cached recommendations."""
        if self.redis:
            try:
                pattern = f"{self.namespace}:rec:*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
            except Exception as e:
                logger.warning(f"Error clearing cache: {e}")
        
        self._memory_cache.clear()
        logger.info("Cache cleared")


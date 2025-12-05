import time
import uuid
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from app.cache import RecommendationCache
from app.feedback_loop import FeedbackLogger
from app.feature_store import FeatureStore
from app.models.ranking import Ranker
from app.models.retrieval import TwoTowerRetrieval


class RealtimeRecommender:
    """Orchestrates retrieval, ranking, feature IO, caching, and feedback for recommendation requests."""

    def __init__(
        self,
        user_seed_path: str = "data/sample_users.json",
        candidate_multiplier: int = 4,
        use_cache: bool = True,
        cache_ttl: int = 300,
        enable_feedback: bool = True,
    ):
        self.retrieval = TwoTowerRetrieval()
        self.ranker = Ranker()
        self.feature_store = FeatureStore()
        self.cache = RecommendationCache(ttl_seconds=cache_ttl) if use_cache else None
        self.feedback_logger = FeedbackLogger() if enable_feedback else None
        self.candidate_multiplier = candidate_multiplier
        self.default_user_features = self._load_user_seed(user_seed_path)
        # Warm store for first-run experience
        self.feature_store.bulk_load_from_file(user_seed_path, overwrite=False)
    def _load_user_seed(self, path: str) -> Dict[str, np.ndarray]:
        try:
            import json

            with open(path) as f:
                payload = json.load(f)
        except FileNotFoundError:
            logger.warning(f"User seed file {path} missing")
            return {}

        seed = {}
        for record in payload:
            user_id = record.get("user_id")
            features = record.get("features")
            if user_id and features:
                seed[user_id] = np.array(features, dtype=np.float32)
        return seed

    def _get_user_features(self, user_id: str) -> np.ndarray | None:
        features = self.feature_store.get_user_features(user_id)
        if features is not None:
            return features
        fallback = self.default_user_features.get(user_id)
        if fallback is not None:
            self.feature_store.set_user_features(user_id, fallback)
        return fallback

    def recommend(
        self,
        user_id: str,
        num_results: int = 10,
        use_cache: bool = True,
        context: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Dict:
        """Generate recommendations with caching and feedback logging."""
        request_id = request_id or str(uuid.uuid4())
        start = time.perf_counter()
        
        # Try cache first
        if use_cache and self.cache:
            cached = self.cache.get(user_id, num_results, context)
            if cached is not None:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.debug(f"Cache hit for user {user_id}")
                
                # Log impression from cache
                if self.feedback_logger:
                    item_ids = [item["item_id"] for item in cached]
                    self.feedback_logger.log_impression(
                        user_id, item_ids, request_id, context
                    )
                
                return {
                    "items": cached,
                    "latency_ms": latency_ms,
                    "cached": True,
                    "request_id": request_id,
                }
        
        # Generate fresh recommendations
        user_features = self._get_user_features(user_id)
        num_candidates = max(num_results * self.candidate_multiplier, num_results)

        candidates = self.retrieval.retrieve(user_features, top_k=num_candidates)
        ranked = self._rerank(user_features, candidates, num_results)

        latency_ms = (time.perf_counter() - start) * 1000
        
        # Cache results
        if use_cache and self.cache:
            self.cache.set(user_id, num_results, ranked, context)
        
        # Log feedback
        if self.feedback_logger:
            item_ids = [item["item_id"] for item in ranked]
            self.feedback_logger.log_impression(user_id, item_ids, request_id, context)
            self.feedback_logger.log_recommendation_request(
                user_id, num_results, latency_ms, request_id, context
            )
        
        return {
            "items": ranked,
            "latency_ms": latency_ms,
            "cached": False,
            "request_id": request_id,
        }

    def _rerank(
        self,
        user_features: np.ndarray | None,
        candidates: List[Dict],
        limit: int,
    ) -> List[Dict]:
        if not candidates:
            return []

        if user_features is None:
            user_features = np.mean(
                [c["features"] for c in candidates], axis=0, dtype=np.float32
            )
            user_features = np.expand_dims(user_features, axis=0)
        else:
            user_features = np.atleast_2d(user_features)

        candidate_features = np.array([cand["features"] for cand in candidates])
        scores = self.ranker.rank(user_features, candidate_features)

        scored = []
        for cand, rank_score in zip(candidates, scores):
            scored.append(
                {
                    "item_id": cand["item_id"],
                    "score": float(rank_score),
                    "metadata": {
                        k: v
                        for k, v in cand["metadata"].items()
                        if k not in {"features"}
                    },
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


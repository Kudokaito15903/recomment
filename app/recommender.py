import time
from typing import Dict, List

import numpy as np
from loguru import logger

from app.feature_store import FeatureStore
from app.models.ranking import Ranker
from app.models.retrieval import TwoTowerRetrieval


class RealtimeRecommender:
    """Orchestrates retrieval, ranking, and feature IO for recommendation requests."""

    def __init__(
        self,
        user_seed_path: str = "data/sample_users.json",
        candidate_multiplier: int = 4,
    ):
        self.retrieval = TwoTowerRetrieval()
        self.ranker = Ranker()
        self.feature_store = FeatureStore()
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

    def recommend(self, user_id: str, num_results: int = 10) -> Dict:
        start = time.perf_counter()
        user_features = self._get_user_features(user_id)
        num_candidates = max(num_results * self.candidate_multiplier, num_results)

        candidates = self.retrieval.retrieve(user_features, top_k=num_candidates)
        ranked = self._rerank(user_features, candidates, num_results)

        latency_ms = (time.perf_counter() - start) * 1000
        return {"items": ranked, "latency_ms": latency_ms}

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


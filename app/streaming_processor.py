"""Streaming processor for event enrichment, feature engineering, and aggregation."""
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from loguru import logger


class EventEnricher:
    """Enriches raw events with additional context and metadata."""

    def __init__(self):
        self.item_metadata = {}
        self.user_profiles = {}

    def load_item_metadata(self, items: List[Dict]) -> None:
        """Load item metadata for enrichment."""
        self.item_metadata = {item["item_id"]: item for item in items}
        logger.info(f"Loaded metadata for {len(self.item_metadata)} items")

    def enrich_event(self, event: Dict) -> Dict:
        """Enrich event with item metadata and computed fields."""
        enriched = event.copy()
        
        # Add timestamp if missing
        if "timestamp" not in enriched:
            enriched["timestamp"] = int(time.time() * 1000)
        
        # Enrich with item metadata
        item_id = event.get("item_id")
        if item_id and item_id in self.item_metadata:
            item_meta = self.item_metadata[item_id]
            enriched["item_category"] = item_meta.get("category", "unknown")
            enriched["item_popularity"] = item_meta.get("popularity", 0.0)
            enriched["item_price"] = item_meta.get("price", 0.0)
        
        # Add session context
        enriched["hour_of_day"] = int(time.strftime("%H"))
        enriched["day_of_week"] = int(time.strftime("%w"))
        
        return enriched


class RealTimeFeatureEngineer:
    """Computes real-time features from event streams."""

    def __init__(self, decay_factor: float = 0.9):
        self.decay_factor = decay_factor
        self.user_stats = defaultdict(lambda: {
            "click_count": 0,
            "view_count": 0,
            "purchase_count": 0,
            "last_interaction_time": 0,
            "item_interactions": defaultdict(int),
            "category_preferences": defaultdict(float),
        })

    def update_user_stats(self, event: Dict) -> None:
        """Update user statistics from event."""
        user_id = event.get("user_id")
        if not user_id:
            return
        
        event_type = event.get("event_type", "").lower()
        timestamp = event.get("timestamp", int(time.time() * 1000))
        
        stats = self.user_stats[user_id]
        
        # Update counts
        if event_type == "click":
            stats["click_count"] += 1
        elif event_type == "view":
            stats["view_count"] += 1
        elif event_type == "purchase":
            stats["purchase_count"] += 1
        
        # Update item interactions
        item_id = event.get("item_id")
        if item_id:
            stats["item_interactions"][item_id] += 1
        
        # Update category preferences (with decay)
        category = event.get("item_category")
        if category:
            # Apply decay to existing preferences
            for cat in stats["category_preferences"]:
                stats["category_preferences"][cat] *= self.decay_factor
            # Add new interaction
            stats["category_preferences"][category] = (
                stats["category_preferences"].get(category, 0) + 1.0
            )
        
        stats["last_interaction_time"] = timestamp

    def compute_user_features(self, user_id: str, base_features: np.ndarray) -> np.ndarray:
        """Compute enhanced user features from real-time stats."""
        stats = self.user_stats.get(user_id, {})
        
        # Compute aggregated features
        total_interactions = (
            stats["click_count"] + stats["view_count"] + stats["purchase_count"]
        )
        click_rate = (
            stats["click_count"] / total_interactions
            if total_interactions > 0
            else 0.0
        )
        purchase_rate = (
            stats["purchase_count"] / total_interactions
            if total_interactions > 0
            else 0.0
        )
        
        # Time since last interaction (normalized)
        time_since_last = (
            (time.time() * 1000 - stats["last_interaction_time"]) / 3600000
            if stats["last_interaction_time"] > 0
            else 24.0
        )
        
        # Top category preference
        top_category_score = (
            max(stats["category_preferences"].values())
            if stats["category_preferences"]
            else 0.0
        )
        
        # Combine with base features
        enhanced_features = np.append(
            base_features,
            [
                total_interactions / 100.0,  # Normalized
                click_rate,
                purchase_rate,
                min(time_since_last / 24.0, 1.0),  # Normalized to [0, 1]
                top_category_score / 10.0,  # Normalized
            ],
        )
        
        return enhanced_features.astype(np.float32)


class StreamingAggregator:
    """Aggregates events for batch processing and feature updates."""

    def __init__(self, window_size_seconds: int = 60):
        self.window_size_seconds = window_size_seconds
        self.window_buffer = defaultdict(list)

    def add_event(self, event: Dict) -> None:
        """Add event to current window."""
        timestamp = event.get("timestamp", int(time.time() * 1000))
        window_key = timestamp // (self.window_size_seconds * 1000)
        self.window_buffer[window_key].append(event)

    def get_window_aggregates(self, window_key: int) -> Dict:
        """Get aggregated statistics for a window."""
        events = self.window_buffer.get(window_key, [])
        if not events:
            return {}
        
        # Aggregate by event type
        event_counts = defaultdict(int)
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        
        for event in events:
            event_type = event.get("event_type", "unknown")
            user_id = event.get("user_id")
            item_id = event.get("item_id")
            
            event_counts[event_type] += 1
            if user_id:
                user_counts[user_id] += 1
            if item_id:
                item_counts[item_id] += 1
        
        return {
            "window_key": window_key,
            "total_events": len(events),
            "event_type_counts": dict(event_counts),
            "unique_users": len(user_counts),
            "unique_items": len(item_counts),
            "top_items": dict(sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        }


class StreamingProcessor:
    """Main streaming processor orchestrating enrichment, feature engineering, and aggregation."""

    def __init__(
        self,
        items: Optional[List[Dict]] = None,
        decay_factor: float = 0.9,
        window_size_seconds: int = 60,
    ):
        self.enricher = EventEnricher()
        self.feature_engineer = RealTimeFeatureEngineer(decay_factor=decay_factor)
        self.aggregator = StreamingAggregator(window_size_seconds=window_size_seconds)
        
        if items:
            self.enricher.load_item_metadata(items)

    def process_event(self, event: Dict) -> Dict:
        """Process a single event through the pipeline."""
        # Enrich event
        enriched = self.enricher.enrich_event(event)
        
        # Update feature engineering stats
        self.feature_engineer.update_user_stats(enriched)
        
        # Add to aggregator
        self.aggregator.add_event(enriched)
        
        return enriched

    def get_user_features(self, user_id: str, base_features: np.ndarray) -> np.ndarray:
        """Get enhanced user features with real-time stats."""
        return self.feature_engineer.compute_user_features(user_id, base_features)

    def get_window_stats(self, window_key: Optional[int] = None) -> Dict:
        """Get aggregated statistics for a window."""
        if window_key is None:
            window_key = int(time.time()) // self.aggregator.window_size_seconds
        return self.aggregator.get_window_aggregates(window_key)


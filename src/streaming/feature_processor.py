"""
Real-time Feature Processing with pandas/NumPy.
Replaces the previous PySpark/Delta pipeline with a lightweight alternative that works
in any standard Python environment.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import structlog
import yaml

from ..data.local_store import LocalDataStore
from .kafka_producer import KafkaConsumer

logger = structlog.get_logger()


class FeatureProcessor:
    """Real-time feature processing built on pandas and scikit-learn."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_store = LocalDataStore(self.config)
        self.kafka_config = self.config.get('streaming', {}).get('kafka', {})
        self.feature_stats: Dict[str, float] = {}
        self.pipeline: Optional[Pipeline] = None
        self.pipeline_trained = False
        self.dimensionality_reduction = 0.67

        feature_cfg = self.config.get('features', {}).get('dimensionality_reduction', {})
        self.target_variance = feature_cfg.get('target_variance', 0.95)
        self.max_components = feature_cfg.get('max_components', 1000)

    # ------------------------------------------------------------------
    def process_events(self, events: Iterable[Dict[str, Any]]) -> None:
        """Process an iterable of interaction events."""
        events = list(events)
        if not events:
            return

        df = pd.DataFrame(events)
        processed = self._prepare_dataframe(df)
        feature_frame = self._build_feature_frame(processed)

        if feature_frame.empty:
            logger.warning("Feature frame is empty; skipping batch")
            return

        self._ensure_pipeline(feature_frame)
        transformed = self.pipeline.transform(feature_frame)

        self._update_user_profiles(processed, transformed)
        self._update_item_features(processed, transformed)
        self._update_feature_stats(feature_frame, transformed)

    def process_existing_dataset(self) -> None:
        """One-off helper to process the current LocalDataStore dataset."""
        interactions = self.data_store.load_interactions()
        if interactions.empty:
            logger.info("No interactions available to process")
            return

        logger.info("Processing %d historical interactions", len(interactions))
        self.process_events(interactions.to_dict(orient='records'))

    # ------------------------------------------------------------------
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        df['hour_of_day'] = df['timestamp'].dt.hour.fillna(0)
        df['day_of_week'] = df['timestamp'].dt.dayofweek.fillna(0)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Session features
        df['session_duration'] = (
            df.groupby('session_id')['timestamp']
            .transform(lambda x: (x.max() - x.min()).total_seconds() if len(x) > 1 else 0)
            .fillna(0)
        )
        df['interactions_in_session'] = df.groupby('session_id')['session_id'].transform('count')

        # User/item historical stats (simple rolling estimates)
        df['user_avg_rating'] = df.groupby('user_id')['rating'].transform('mean')
        df['item_avg_rating'] = df.groupby('item_id')['rating'].transform('mean')
        df['user_interaction_count'] = df.groupby('user_id')['user_id'].transform('count')
        df['item_interaction_count'] = df.groupby('item_id')['item_id'].transform('count')

        # Normalize missing data
        numeric_cols = [
            'rating', 'hour_of_day', 'day_of_week', 'session_duration',
            'interactions_in_session', 'user_avg_rating', 'item_avg_rating',
            'user_interaction_count', 'item_interaction_count'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _build_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [
            'rating',
            'hour_of_day',
            'day_of_week',
            'is_weekend',
            'session_duration',
            'interactions_in_session',
            'user_avg_rating',
            'item_avg_rating',
            'user_interaction_count',
            'item_interaction_count',
        ]
        return df[feature_columns]

    def _ensure_pipeline(self, feature_frame: pd.DataFrame) -> None:
        if self.pipeline is not None and self.pipeline_trained:
            return

        n_features = feature_frame.shape[1]
        target_components = max(1, int(n_features * (1 - self.dimensionality_reduction)))
        target_components = min(target_components, self.max_components)

        self.pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=target_components))
            ]
        )
        self.pipeline.fit(feature_frame)
        self.pipeline_trained = True

        variance = np.sum(self.pipeline.named_steps['pca'].explained_variance_ratio_)
        logger.info(
            "Trained feature pipeline (%d -> %d dims, variance %.3f)",
            n_features,
            target_components,
            variance,
        )

    # ------------------------------------------------------------------
    def _update_user_profiles(self, df: pd.DataFrame, transformed: np.ndarray) -> None:
        components = pd.DataFrame(
            transformed,
            columns=[f'pca_{i}' for i in range(transformed.shape[1])]
        )
        enriched = pd.concat([df[['user_id', 'timestamp', 'rating']], components], axis=1)

        user_profiles = (
            enriched
            .groupby('user_id')
                .agg(
                avg_rating=('rating', 'mean'),
                interaction_count=('rating', 'count'),
                last_interaction=('timestamp', 'max'),
                **{col: (col, 'mean') for col in components.columns}
            )
            .reset_index()
        )

        self.data_store.save_user_profiles(user_profiles)
        logger.debug("Updated %d user profiles", len(user_profiles))

    def _update_item_features(self, df: pd.DataFrame, transformed: np.ndarray) -> None:
        components = pd.DataFrame(
            transformed,
            columns=[f'pca_{i}' for i in range(transformed.shape[1])]
        )
        enriched = pd.concat([df[['item_id', 'timestamp', 'rating']], components], axis=1)

        item_features = (
            enriched
            .groupby('item_id')
                .agg(
                avg_rating=('rating', 'mean'),
                interaction_count=('rating', 'count'),
                last_interaction=('timestamp', 'max'),
                **{col: (col, 'mean') for col in components.columns}
            )
            .reset_index()
        )

        self.data_store.save_item_features(item_features)
        logger.debug("Updated %d item feature rows", len(item_features))

    def _update_feature_stats(self, feature_frame: pd.DataFrame, transformed: np.ndarray) -> None:
        summary = feature_frame.describe().to_dict()
        for column, metrics in summary.items():
            for stat, value in metrics.items():
                self.feature_stats[f"{column}_{stat}"] = float(value)

        self.feature_stats['pca_variance'] = float(
            np.sum(self.pipeline.named_steps['pca'].explained_variance_ratio_)
        )
        self.feature_stats['last_update_ts'] = time.time()

    # ------------------------------------------------------------------
    def get_feature_stats(self) -> Dict[str, Any]:
        return {
            'feature_stats': self.feature_stats,
            'pipeline_trained': self.pipeline_trained,
            'd_reduction_rate': self.dimensionality_reduction,
            'target_variance': self.target_variance,
        }

    async def start_streaming(self) -> None:
        """Optional helper to consume events from Kafka using kafka-python."""
        topics = [self.kafka_config.get('topics', {}).get('user_interactions', 'user_interactions')]
        consumer = KafkaConsumer(self.kafka_config, topics)

        async def handle_message(**message):
            value = message.get('value', {})
            await asyncio.get_event_loop().run_in_executor(
                None, self.process_events, [value]
            )

        logger.info("Starting Kafka consumer for feature processing...")
        await consumer.start_consuming(handle_message)


# Example usage ---------------------------------------------------------------
async def main():
    processor = FeatureProcessor()
    try:
        processor.process_existing_dataset()
        await processor.start_streaming()
    except KeyboardInterrupt:
        logger.info("Stopping feature processor...")
    except Exception as exc:
        logger.error("Feature processor error: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())

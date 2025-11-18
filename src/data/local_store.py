"""
Utility helpers for working with local (pandas-based) datasets.

This module replaces the previous PySpark/Delta Lake layer with lightweight
CSV/Parquet helpers that run anywhere a standard Python data stack is available.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd


class LocalDataStore:
    """Simple wrapper around pandas for reading/writing project datasets."""

    def __init__(self, config: Dict[str, Any]):
        data_config = config.get('data_sources', {})

        self.interactions_path = Path(
            data_config.get('interactions', 'data/sample_interactions.csv')
        )
        self.items_path = Path(
            data_config.get('items', 'data/sample_items.csv')
        )
        self.user_profiles_path = Path(
            data_config.get('user_profiles', 'data/user_profiles.csv')
        )
        self.item_features_path = Path(
            data_config.get('item_features', 'data/item_features.csv')
        )

        for path in (
            self.interactions_path,
            self.items_path,
            self.user_profiles_path,
            self.item_features_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generic helpers
    def _read_table(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()

        suffix = path.suffix.lower()
        try:
            if suffix in ('.csv', '.txt'):
                return pd.read_csv(path)
            if suffix in ('.parquet', '.pq'):
                return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Pandas could not read {path} because optional dependency "
                f"is missing: {exc}. Install 'pyarrow' to use Parquet files."
            ) from exc

        raise ValueError(f"Unsupported file format for {path}")

    def _write_table(self, df: pd.DataFrame, path: Path) -> None:
        suffix = path.suffix.lower()

        if suffix in ('.csv', '.txt'):
            df.to_csv(path, index=False)
            return
        if suffix in ('.parquet', '.pq'):
            try:
                df.to_parquet(path, index=False)
                return
            except ImportError as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Pandas could not write {path} because optional dependency "
                    f"is missing: {exc}. Install 'pyarrow' to use Parquet files."
                ) from exc

        raise ValueError(f"Unsupported file format for {path}")

    # ------------------------------------------------------------------
    # Public API
    def load_interactions(self) -> pd.DataFrame:
        """Load user-item interactions with sensible defaults."""
        df = self._read_table(self.interactions_path)
        if df.empty:
            return df

        # Ensure consistent column casing/order
        expected_columns = [
            'user_id',
            'item_id',
            'rating',
            'timestamp',
            'interaction_type',
            'session_id',
        ]
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Interaction dataset is missing columns: {missing}"
            )

        return df.sort_values('timestamp').reset_index(drop=True)

    def append_interaction(self, interaction: Dict[str, Any]) -> None:
        """Append a single interaction (in-memory) and persist it."""
        interaction = interaction.copy()
        interaction.setdefault('timestamp', time.time())
        interaction.setdefault('interaction_type', 'rating')
        interaction.setdefault('session_id', f"session_{int(time.time())}")

        existing = self.load_interactions()
        updated = pd.concat(
            [existing, pd.DataFrame([interaction])],
            ignore_index=True,
        )
        self._write_table(updated, self.interactions_path)

    def load_items(self) -> pd.DataFrame:
        """Load the item catalog."""
        df = self._read_table(self.items_path)
        if df.empty:
            raise FileNotFoundError(
                f"Item catalog not found at {self.items_path}. "
                "Provide data or update config.data_sources.items."
            )
        return df

    def load_user_profiles(self) -> pd.DataFrame:
        return self._read_table(self.user_profiles_path)

    def save_user_profiles(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        existing = self.load_user_profiles()
        combined = self._merge_by_key(existing, df, key='user_id')
        self._write_table(combined, self.user_profiles_path)

    def load_item_features(self) -> pd.DataFrame:
        return self._read_table(self.item_features_path)

    def save_item_features(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        existing = self.load_item_features()
        combined = self._merge_by_key(existing, df, key='item_id')
        self._write_table(combined, self.item_features_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_by_key(
        existing: pd.DataFrame,
        incoming: pd.DataFrame,
        key: str,
    ) -> pd.DataFrame:
        if existing.empty:
            return incoming.reset_index(drop=True)

        combined = (
            pd.concat([existing, incoming])
            .sort_values(by=[key, 'timestamp'] if 'timestamp' in incoming.columns else key)
            .drop_duplicates(subset=[key], keep='last')
        )
        return combined.reset_index(drop=True)


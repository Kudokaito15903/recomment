import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class UserTower(nn.Module):
    def __init__(self, input_dim: int = 16, embed_dim: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, input_dim: int = 16, embed_dim: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x), dim=-1)


class TwoTowerRetrieval:
    """Vector similarity search with optional trained towers and FAISS index."""

    def __init__(
        self,
        items_path: str | Path = "data/sample_items.json",
        model_dir: str | Path = "models",
        embed_dim: int = 32,
    ):
        self.items_path = Path(items_path)
        self.model_dir = Path(model_dir)
        self.embed_dim = embed_dim

        self.items = self._load_items()
        self.item_ids = [item["item_id"] for item in self.items]
        self.item_features = np.array(
            [item["features"] for item in self.items], dtype=np.float32
        )
        self.item_meta: Dict[str, Dict] = {item["item_id"]: item for item in self.items}

        input_dim = self.item_features.shape[1]
        self.user_tower = UserTower(input_dim=input_dim, embed_dim=embed_dim)
        self.item_tower = ItemTower(input_dim=input_dim, embed_dim=embed_dim)

        self._load_trained_weights()
        self._build_or_load_index()
        logger.info(f"TwoTowerRetrieval initialized with {len(self.items)} items")

    def _load_items(self) -> List[Dict]:
        if not self.items_path.exists():
            raise FileNotFoundError(
                f"Item catalog not found at {self.items_path}. "
                "Run scripts/generate_sample_data.py first."
            )
        with self.items_path.open() as f:
            return json.load(f)

    def _load_trained_weights(self) -> None:
        user_path = self.model_dir / "retrieval_user.pt"
        item_path = self.model_dir / "retrieval_item.pt"
        for tower, path in ((self.user_tower, user_path), (self.item_tower, item_path)):
            if path.exists():
                state = torch.load(path, map_location="cpu")
                tower.load_state_dict(state)
                logger.info(f"Loaded tower weights from {path}")

    def _build_or_load_index(self) -> None:
        index_path = self.model_dir / "item_index.faiss"
        embeddings_path = self.model_dir / "item_embeddings.npy"
        dim = self.embed_dim
        self.index = faiss.IndexFlatIP(dim)

        if index_path.exists() and embeddings_path.exists():
            self.item_embeddings = np.load(embeddings_path)
            self.index = faiss.read_index(str(index_path))
            logger.info("Loaded FAISS index from disk")
            return

        with torch.no_grad():
            item_tensor = torch.tensor(self.item_features, dtype=torch.float32)
            self.item_embeddings = self.item_tower(item_tensor).numpy()

        faiss.normalize_L2(self.item_embeddings)
        self.index.add(self.item_embeddings)

        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.item_embeddings)
        faiss.write_index(self.index, str(index_path))
        logger.info("Built FAISS index from scratch")

    def refresh_index(self, items: List[Dict] | None = None) -> None:
        if items is not None:
            self.items = items
            self.item_ids = [item["item_id"] for item in self.items]
            self.item_features = np.array(
                [item["features"] for item in self.items], dtype=np.float32
            )
            self.item_meta = {item["item_id"]: item for item in self.items}
        self._build_or_load_index()

    def encode_user(self, user_features: np.ndarray) -> np.ndarray:
        if user_features.ndim == 1:
            user_features = np.expand_dims(user_features, axis=0)
        with torch.no_grad():
            tensor = torch.tensor(user_features, dtype=torch.float32)
            embedding = self.user_tower(tensor).numpy()
        faiss.normalize_L2(embedding)
        return embedding

    def retrieve(self, user_features: np.ndarray | None, top_k: int = 20) -> List[Dict]:
        if user_features is None:
            return self._popular_fallback(top_k)

        query = self.encode_user(user_features)
        distances, indices = self.index.search(query, top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            item_id = self.item_ids[idx]
            results.append(
                {
                    "item_id": item_id,
                    "score": float(score),
                    "features": self.item_features[idx],
                    "metadata": self.item_meta[item_id],
                }
            )
        return results

    def _popular_fallback(self, top_k: int) -> List[Dict]:
        sorted_items = sorted(
            self.items, key=lambda item: item.get("popularity", 0), reverse=True
        )
        fallback = []
        for item in sorted_items[:top_k]:
            fallback.append(
                {
                    "item_id": item["item_id"],
                    "score": float(item.get("popularity", 0)),
                    "features": np.array(item["features"], dtype=np.float32),
                    "metadata": item,
                }
            )
        return fallback

    def get_item_features(self, item_id: str) -> np.ndarray | None:
        idx = self.item_ids.index(item_id) if item_id in self.item_ids else -1
        if idx == -1:
            return None
        return self.item_features[idx]

    def get_item_metadata(self, item_id: str) -> Dict | None:
        return self.item_meta.get(item_id)

# train_two_tower.py
import json
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from app.models.retrieval import ItemTower, UserTower  # noqa: E402

# -----------------------
# Utilities / Dataset
# -----------------------
class TwoTowerDataset(Dataset):
    def __init__(self, user_vectors, item_vectors, labels):
        self.user_vectors = torch.tensor(user_vectors, dtype=torch.float32)
        self.item_vectors = torch.tensor(item_vectors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_vectors[idx], self.item_vectors[idx], self.labels[idx]


def build_dataset(
    events,
    user_map,
    item_map,
    negatives_per_positive: int = 2,
    ensure_not_positive: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create (user_vector, item_vector, label) pairs with negative sampling.
    negatives_per_positive: how many negatives for each positive example.
    ensure_not_positive: try to avoid selecting the same item as positive when sampling negatives.
    """
    pairs = []
    labels = []
    item_ids = list(item_map.keys())
    for event in events:
        user = user_map.get(event["user_id"])
        item = item_map.get(event["item_id"])
        if user is None or item is None:
            continue

        # positive
        pairs.append((user, item))
        labels.append(1.0)

        # negatives
        for _ in range(negatives_per_positive):
            # sample until we get a different item (or give up after few tries)
            for _try in range(5):
                neg_item_id = random.choice(item_ids)
                if not ensure_not_positive or neg_item_id != event["item_id"]:
                    break
            neg_item = item_map[neg_item_id]
            pairs.append((user, neg_item))
            labels.append(0.0)

    if len(pairs) == 0:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,))

    user_vectors = np.stack([p[0] for p in pairs])
    item_vectors = np.stack([p[1] for p in pairs])
    return user_vectors, item_vectors, np.array(labels, dtype=np.float32)


# -----------------------
# Training entrypoint
# -----------------------
def train(
    num_epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    negatives_per_positive: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
    use_normalize_embeddings: bool = True,
    temperature: float = 0.1,
):
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = BASE_DIR / "data"
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    users = json.loads((data_dir / "sample_users.json").read_text())
    items = json.loads((data_dir / "sample_items.json").read_text())
    events = json.loads((data_dir / "sample_events.json").read_text())

    user_map = {u["user_id"]: np.array(u["features"], dtype=np.float32) for u in users}
    item_map = {i["item_id"]: np.array(i["features"], dtype=np.float32) for i in items}

    user_vectors, item_vectors, labels = build_dataset(
        events, user_map, item_map, negatives_per_positive=negatives_per_positive
    )

    if user_vectors.size == 0:
        print("No training pairs constructed. Exiting.")
        return

    # Dataset + split
    dataset = TwoTowerDataset(user_vectors, item_vectors, labels)
    if 0.0 < val_fraction < 1.0:
        val_size = int(len(dataset) * val_fraction)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds = dataset
        val_ds = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda")
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
        if val_ds is not None
        else None
    )

    input_dim = item_vectors.shape[1]
    user_tower = UserTower(input_dim=input_dim).to(device)
    item_tower = ItemTower(input_dim=input_dim).to(device)

    optimizer = optim.Adam(
        list(user_tower.parameters()) + list(item_tower.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    bce_loss = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        user_tower.train()
        item_tower.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for user_batch, item_batch, label_batch in train_loader:
            user_batch = user_batch.to(device)
            item_batch = item_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()

            u_emb = user_tower(user_batch)  # (B, D)
            v_emb = item_tower(item_batch)  # (B, D)

            if use_normalize_embeddings:
                u_emb = F.normalize(u_emb, p=2, dim=-1)
                v_emb = F.normalize(v_emb, p=2, dim=-1)

            # dot product
            logits = torch.sum(u_emb * v_emb, dim=-1) / temperature  # (B,)

            loss = bce_loss(logits, label_batch)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                epoch_correct += (preds == label_batch).sum().item()
                epoch_total += label_batch.numel()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total if epoch_total else 0.0

        # Validation
        if val_loader is not None:
            user_tower.eval()
            item_tower.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for user_batch, item_batch, label_batch in val_loader:
                    user_batch = user_batch.to(device)
                    item_batch = item_batch.to(device)
                    label_batch = label_batch.to(device)

                    u_emb = user_tower(user_batch)
                    v_emb = item_tower(item_batch)

                    if use_normalize_embeddings:
                        u_emb = F.normalize(u_emb, p=2, dim=-1)
                        v_emb = F.normalize(v_emb, p=2, dim=-1)

                    logits = torch.sum(u_emb * v_emb, dim=-1) / temperature
                    loss = bce_loss(logits, label_batch)

                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (preds == label_batch).sum().item()
                    val_total += label_batch.numel()
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total if val_total else 0.0
            print(
                f"Epoch {epoch}/{num_epochs} | train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} "
                f"| val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}"
            )

            # scheduler step
            scheduler.step(avg_val_loss)

            # save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                ckpt = {
                    "epoch": epoch,
                    "user_state": user_tower.state_dict(),
                    "item_state": item_tower.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }
                torch.save(ckpt, model_dir / "best_retrieval_ckpt.pt")
                print(f"Saved best checkpoint (epoch {epoch}, val_loss={avg_val_loss:.4f})")
        else:
            print(f"Epoch {epoch}/{num_epochs} | train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f}")
            # step scheduler with train loss if no val
            scheduler.step(avg_train_loss)

        # regular checkpoint each epoch
        torch.save(user_tower.state_dict(), model_dir / "retrieval_user.pt")
        torch.save(item_tower.state_dict(), model_dir / "retrieval_item.pt")

    print(f"Training finished. Best epoch: {best_epoch} val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    train()

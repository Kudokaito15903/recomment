import json
import pickle
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


def build_training_matrix(events, users, items, negatives_per_positive=1):
    user_map = {u["user_id"]: np.array(u["features"], dtype=np.float32) for u in users}
    item_map = {i["item_id"]: np.array(i["features"], dtype=np.float32) for i in items}
    features = []
    labels = []

    for event in events:
        user_vec = user_map.get(event["user_id"])
        item_vec = item_map.get(event["item_id"])
        if user_vec is None or item_vec is None:
            continue

        label = 1 if event["event_type"] in {"click", "purchase"} else 0
        features.append(np.hstack([user_vec, item_vec]))
        labels.append(label)

        for _ in range(negatives_per_positive):
            neg_item_id = random.choice(list(item_map.keys()))
            neg_vec = item_map[neg_item_id]
            features.append(np.hstack([user_vec, neg_vec]))
            labels.append(0)

    return np.vstack(features), np.array(labels)


def train():
    data_dir = BASE_DIR / "data"
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    users = json.loads((data_dir / "sample_users.json").read_text())
    items = json.loads((data_dir / "sample_items.json").read_text())
    events = json.loads((data_dir / "sample_events.json").read_text())

    X, y = build_training_matrix(events, users, items)
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
    model.fit(X, y)

    with open(model_dir / "ranker.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved ranking model to {model_dir / 'ranker.pkl'}")


if __name__ == "__main__":
    train()

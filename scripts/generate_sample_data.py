import json
import numpy as np
import random
import os

os.makedirs("data", exist_ok=True)

# --- 1. Items ---
num_items = 50
item_dim = 16  # số chiều feature / embedding

items = []
categories = ["shoes", "electronics", "clothes", "books"]

for i in range(1, num_items+1):
    item = {
        "item_id": f"p_{i}",
        "features": np.random.rand(item_dim).tolist(),
        "category": random.choice(categories),
        "price": round(random.uniform(5, 500), 2),
        "popularity": random.randint(1, 1000)
    }
    items.append(item)

with open("data/sample_items.json", "w") as f:
    json.dump(items, f, indent=2)
print(f"Generated {num_items} items in data/sample_items.json")

# --- 2. Users ---
num_users = 10
user_dim = item_dim  # same dimension for simplicity

users = []
for i in range(1, num_users+1):
    user = {
        "user_id": f"u_{i}",
        "features": np.random.rand(user_dim).tolist(),
        "age": random.randint(18, 50),
        "country": random.choice(["VN", "US", "JP"]),
        "recent_item_ids": [f"p_{random.randint(1, num_items)}" for _ in range(5)]
    }
    users.append(user)

with open("data/sample_users.json", "w") as f:
    json.dump(users, f, indent=2)
print(f"Generated {num_users} users in data/sample_users.json")

# --- 3. Events ---
num_events = 100
event_types = ["view", "click", "purchase"]

events = []
for i in range(num_events):
    event = {
        "user_id": random.choice([u["user_id"] for u in users]),
        "item_id": random.choice([item["item_id"] for item in items]),
        "event_type": random.choice(event_types),
        "timestamp": int(np.random.randint(1700000000, 1700500000) * 1000)
    }
    events.append(event)

with open("data/sample_events.json", "w") as f:
    json.dump(events, f, indent=2)
print(f"Generated {num_events} events in data/sample_events.json")

from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

user_ids = ["u_1", "u_2", "u_3"]
item_ids = ["p_1", "p_2", "p_3"]

for i in range(10):
    event = {
        "user_id": random.choice(user_ids),
        "event_type": "view",
        "item_id": random.choice(item_ids),
        "timestamp": int(time.time() * 1000)
    }
    producer.send("user-events", value=event)
    print(f"Sent event: {event}")
    time.sleep(0.5)

producer.flush()

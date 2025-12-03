from prometheus_client import Counter, Histogram
from loguru import logger

REQ_COUNTER = Counter("recommender_requests_total", "Total requests")
REQ_LATENCY = Histogram("recommender_request_latency_seconds", "Request latency")

def log_request(user_id, latency_sec):
    logger.info(f"user_id={user_id} latency={latency_sec:.4f}s")
    REQ_COUNTER.inc()
    REQ_LATENCY.observe(latency_sec)

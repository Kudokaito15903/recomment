"""Feedback loop for logging events back to stream after serving recommendations."""
import json
import os
import time
from typing import Dict, List, Optional

from kafka import KafkaProducer
from loguru import logger


class FeedbackLogger:
    """Logs recommendation feedback events back to stream for online learning."""

    def __init__(
        self,
        kafka_bootstrap_servers: Optional[str] = None,
        feedback_topic: str = "recommendation-feedback",
        use_kafka: bool = True,
    ):
        self.feedback_topic = feedback_topic
        self.use_kafka = use_kafka
        self.events_buffer = []
        
        if use_kafka:
            bootstrap_servers = (
                kafka_bootstrap_servers
                or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            )
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",  # Wait for all replicas
                    retries=3,
                )
                logger.info(f"Feedback logger connected to Kafka at {bootstrap_servers}")
            except Exception as e:
                logger.warning(f"Kafka producer unavailable ({e}); using buffer only")
                self.use_kafka = False
                self.producer = None
        else:
            self.producer = None

    def log_impression(
        self,
        user_id: str,
        item_ids: List[str],
        request_id: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> None:
        """Log impression event when recommendations are served."""
        event = {
            "event_type": "impression",
            "user_id": user_id,
            "item_ids": item_ids,
            "timestamp": int(time.time() * 1000),
            "request_id": request_id,
            "context": context or {},
        }
        self._send_event(event)

    def log_interaction(
        self,
        user_id: str,
        item_id: str,
        interaction_type: str,  # click, purchase, view, etc.
        request_id: Optional[str] = None,
        position: Optional[int] = None,
        context: Optional[Dict] = None,
    ) -> None:
        """Log user interaction with recommended item."""
        event = {
            "event_type": interaction_type,
            "user_id": user_id,
            "item_id": item_id,
            "timestamp": int(time.time() * 1000),
            "request_id": request_id,
            "position": position,  # Position in recommendation list
            "context": context or {},
        }
        self._send_event(event)

    def log_recommendation_request(
        self,
        user_id: str,
        num_results: int,
        latency_ms: float,
        request_id: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> None:
        """Log recommendation request for analytics."""
        event = {
            "event_type": "recommendation_request",
            "user_id": user_id,
            "num_results": num_results,
            "latency_ms": latency_ms,
            "timestamp": int(time.time() * 1000),
            "request_id": request_id,
            "context": context or {},
        }
        self._send_event(event)

    def _send_event(self, event: Dict) -> None:
        """Send event to Kafka or buffer."""
        if self.use_kafka and self.producer:
            try:
                future = self.producer.send(self.feedback_topic, event)
                # Don't wait for result to avoid blocking
                future.add_errback(
                    lambda e: logger.error(f"Failed to send feedback event: {e}")
                )
            except Exception as e:
                logger.error(f"Error sending feedback event to Kafka: {e}")
                # Fallback to buffer
                self.events_buffer.append(event)
        else:
            # Buffer events if Kafka unavailable
            self.events_buffer.append(event)
            if len(self.events_buffer) > 1000:
                logger.warning(f"Feedback buffer size: {len(self.events_buffer)}")

    def flush(self) -> None:
        """Flush buffered events (for testing or graceful shutdown)."""
        if self.events_buffer and self.producer:
            for event in self.events_buffer:
                try:
                    self.producer.send(self.feedback_topic, event)
                except Exception as e:
                    logger.error(f"Error flushing feedback event: {e}")
            self.events_buffer.clear()

    def get_buffered_events(self) -> List[Dict]:
        """Get buffered events (for testing or fallback processing)."""
        return self.events_buffer.copy()


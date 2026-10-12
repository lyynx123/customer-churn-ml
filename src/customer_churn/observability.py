"""Lightweight inference observability for Customer Churn Prediction.

Provides structured metrics collection for production inference monitoring
without introducing external dependencies or global state.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ErrorCategory(str, Enum):
    """Explicit error categories for inference failures."""

    VALIDATION_ERROR = "validation_error"
    INFERENCE_ERROR = "inference_error"
    ARTIFACT_INTEGRITY_ERROR = "artifact_integrity_error"
    ARTIFACT_LOAD_ERROR = "artifact_load_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass(frozen=True)
class MetricsEvent:
    """Structured metrics event for a single inference call or batch."""

    event_type: str
    timestamp: str
    invocation_count: int
    prediction_count: int
    positive_predictions: int
    negative_predictions: int
    latency_ms: float
    threshold: float
    model_artifact_hash: str | None
    success: bool
    error_category: str | None
    batch_size: int | None = None


class MetricsCollector:
    """In-memory metrics collector for inference observability.

    Collects and aggregates metrics across multiple inference calls.
    Thread-compatible but not thread-safe; use one collector per process
    or synchronize externally if needed.
    """

    def __init__(
        self,
        model_artifact_hash: str | None = None,
        default_threshold: float | None = None,
    ):
        self._events: list[MetricsEvent] = []
        self._invocation_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._validation_failure_count = 0
        self._inference_failure_count = 0
        self._artifact_integrity_failure_count = 0
        self._artifact_load_failure_count = 0
        self._total_predictions = 0
        self._positive_predictions = 0
        self._negative_predictions = 0
        self._model_artifact_hash = model_artifact_hash
        self._default_threshold = default_threshold

    @property
    def events(self) -> tuple[MetricsEvent, ...]:
        return tuple(self._events)

    @property
    def invocation_count(self) -> int:
        return self._invocation_count

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def validation_failure_count(self) -> int:
        return self._validation_failure_count

    @property
    def inference_failure_count(self) -> int:
        return self._inference_failure_count

    @property
    def artifact_integrity_failure_count(self) -> int:
        return self._artifact_integrity_failure_count

    @property
    def artifact_load_failure_count(self) -> int:
        return self._artifact_load_failure_count

    @property
    def total_predictions(self) -> int:
        return self._total_predictions

    @property
    def positive_predictions(self) -> int:
        return self._positive_predictions

    @property
    def negative_predictions(self) -> int:
        return self._negative_predictions

    @property
    def model_artifact_hash(self) -> str | None:
        return self._model_artifact_hash

    @property
    def default_threshold(self) -> float | None:
        return self._default_threshold

    def record_success(
        self,
        event_type: str,
        prediction_count: int,
        positive_predictions: int,
        negative_predictions: int,
        latency_ms: float,
        threshold: float,
        batch_size: int | None = None,
    ) -> MetricsEvent:
        """Record a successful inference event."""
        self._invocation_count += 1
        self._success_count += 1
        self._total_predictions += prediction_count
        self._positive_predictions += positive_predictions
        self._negative_predictions += negative_predictions

        event = MetricsEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC).isoformat(),
            invocation_count=self._invocation_count,
            prediction_count=prediction_count,
            positive_predictions=positive_predictions,
            negative_predictions=negative_predictions,
            latency_ms=latency_ms,
            threshold=threshold,
            model_artifact_hash=self._model_artifact_hash,
            success=True,
            error_category=None,
            batch_size=batch_size,
        )
        self._events.append(event)
        return event

    def record_failure(
        self,
        event_type: str,
        error_category: ErrorCategory,
        threshold: float,
        latency_ms: float = 0.0,
        batch_size: int | None = None,
    ) -> MetricsEvent:
        """Record a failed inference event."""
        self._invocation_count += 1
        self._failure_count += 1

        if error_category == ErrorCategory.VALIDATION_ERROR:
            self._validation_failure_count += 1
        elif error_category == ErrorCategory.INFERENCE_ERROR:
            self._inference_failure_count += 1
        elif error_category == ErrorCategory.ARTIFACT_INTEGRITY_ERROR:
            self._artifact_integrity_failure_count += 1
        elif error_category == ErrorCategory.ARTIFACT_LOAD_ERROR:
            self._artifact_load_failure_count += 1

        event = MetricsEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC).isoformat(),
            invocation_count=self._invocation_count,
            prediction_count=0,
            positive_predictions=0,
            negative_predictions=0,
            latency_ms=latency_ms,
            threshold=threshold,
            model_artifact_hash=self._model_artifact_hash,
            success=False,
            error_category=error_category.value,
            batch_size=batch_size,
        )
        self._events.append(event)
        return event

    def get_summary(self) -> dict[str, Any]:
        """Return aggregated metrics summary."""
        return {
            "invocation_count": self._invocation_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "validation_failure_count": self._validation_failure_count,
            "inference_failure_count": self._inference_failure_count,
            "artifact_integrity_failure_count": self._artifact_integrity_failure_count,
            "artifact_load_failure_count": self._artifact_load_failure_count,
            "total_predictions": self._total_predictions,
            "positive_predictions": self._positive_predictions,
            "negative_predictions": self._negative_predictions,
            "model_artifact_hash": self._model_artifact_hash,
            "default_threshold": self._default_threshold,
        }

    def write_jsonl(self, path: Path) -> None:
        """Write events to JSON Lines file (append mode).

        Args:
            path: Output file path. File is created if it doesn't exist.

        Raises:
            OSError: If file cannot be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(asdict(event), ensure_ascii=False))
                f.write("\n")

    def clear(self) -> None:
        """Clear all collected events and counters."""
        self._events.clear()
        self._invocation_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._validation_failure_count = 0
        self._inference_failure_count = 0
        self._artifact_integrity_failure_count = 0
        self._artifact_load_failure_count = 0
        self._total_predictions = 0
        self._positive_predictions = 0
        self._negative_predictions = 0


@contextmanager
def metrics_context(
    collector: MetricsCollector,
    event_type: str,
    threshold: float,
    batch_size: int | None = None,
):
    """Context manager for timing and recording inference metrics.

    Usage:
        with metrics_context(collector, "predict_batch", threshold=0.28, batch_size=100):
            result = predictor.predict_batch(X)

    On success: records prediction counts and latency.
    On exception: records failure with appropriate error category.
    """
    start_time = time.perf_counter()
    error_category: ErrorCategory | None = None
    try:
        yield
        latency_ms = (time.perf_counter() - start_time) * 1000
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        error_category = _categorize_error(e)
        raise
    finally:
        if error_category is not None:
            collector.record_failure(
                event_type=event_type,
                error_category=error_category,
                threshold=threshold,
                latency_ms=latency_ms,
                batch_size=batch_size,
            )


def _categorize_error(exception: BaseException) -> ErrorCategory:
    """Categorize an exception into an explicit error category.

    Only categorizes known exception types; unknown exceptions
    are classified as UNKNOWN_ERROR.
    """
    from customer_churn.portable_predict import PortableValidationError

    if isinstance(exception, PortableValidationError):
        return ErrorCategory.VALIDATION_ERROR
    if isinstance(exception, FileNotFoundError):
        return ErrorCategory.ARTIFACT_LOAD_ERROR
    if isinstance(exception, ValueError):
        msg = str(exception).lower()
        if "integrity" in msg or "sha256" in msg:
            return ErrorCategory.ARTIFACT_INTEGRITY_ERROR
        return ErrorCategory.ARTIFACT_LOAD_ERROR
    if isinstance(exception, (OSError, json.JSONDecodeError)):
        return ErrorCategory.ARTIFACT_LOAD_ERROR
    return ErrorCategory.UNKNOWN_ERROR


def create_collector_from_metadata(
    metadata: Any,
) -> MetricsCollector:
    """Create a MetricsCollector populated from InferenceMetadata.

    Args:
        metadata: InferenceMetadata instance with artifact hash and threshold.

    Returns:
        Configured MetricsCollector instance.
    """
    artifact_hash = None
    threshold = None
    if hasattr(metadata, "artifact_integrity"):
        artifact_hash = metadata.artifact_integrity.get("pipeline_sha256")
    if hasattr(metadata, "production_threshold"):
        threshold = metadata.production_threshold
    return MetricsCollector(
        model_artifact_hash=artifact_hash,
        default_threshold=threshold,
    )
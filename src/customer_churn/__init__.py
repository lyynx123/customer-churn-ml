# src/customer_churn/__init__.py
"""Customer churn package initialization."""

from .contract import DataContract, ValidationError, ValidationResult
from .observability import (
    ErrorCategory,
    MetricsCollector,
    MetricsEvent,
    create_collector_from_metadata,
    metrics_context,
)
from .portable_predict import (
    InferenceMetadata,
    PortableChurnPredictor,
    PortableDataContract,
)

__all__ = [
    "DataContract",
    "ErrorCategory",
    "InferenceMetadata",
    "MetricsCollector",
    "MetricsEvent",
    "PortableChurnPredictor",
    "PortableDataContract",
    "ValidationError",
    "ValidationResult",
    "create_collector_from_metadata",
    "metrics_context",
]

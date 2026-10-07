# src/customer_churn/__init__.py
"""Customer churn package initialization."""

from .contract import DataContract, ValidationError, ValidationResult

__all__ = ["DataContract", "ValidationError", "ValidationResult"]

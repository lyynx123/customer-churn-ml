"""Data Contract for production inference boundary.

This module provides a validation layer that sits between raw inference input
and the production pipeline. It validates input against the production schema
derived from the fitted pipeline, enforces strict contracts, and ensures
the production boundary is explicit and auditable.

The contract:
- Uses the EXACT pipeline instance owned by ChurnPredictor
- Derives categorical allowed values from the fitted OneHotEncoder
- Rejects invalid input at the boundary (fail-fast)
- Does NOT perform imputation, fitting, or statistical transformation
- Does NOT load a second pipeline instance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Production threshold from Phase 11 business cost optimization
PRODUCTION_THRESHOLD = 0.28


class ValidationError(Exception):
    """Raised when input validation fails at the production boundary."""

    def __init__(self, message: str, errors: list[str]):
        self.errors = errors
        full_message = message + ": " + "; ".join(errors) if errors else message
        super().__init__(full_message)


@dataclass(frozen=True)
class ValidationResult:
    """Result of successful validation.

    Attributes:
        validated_data: Input data as a DataFrame with validated columns in the
            exact order expected by the production preprocessor.
    """
    validated_data: pd.DataFrame


class DataContract:
    """Production data contract for the churn prediction pipeline.

    The contract validates input data against the production schema derived
    from the fitted production pipeline. It enforces:
    - All required features present
    - No forbidden fields (customerID, Churn)
    - Numeric types, finite values, domain constraints
    - Categorical values exactly matching the fitted OneHotEncoder categories
    - No missing values
    - No unexpected fields

    The contract uses the EXACT pipeline instance from ChurnPredictor and
    derives the categorical domain from the fitted OneHotEncoder.categories_.
    """

    @classmethod
    def _default_numeric_constraints(cls) -> dict[str, dict[str, Any]]:
        return {
            "SeniorCitizen": {"type": "int", "allowed": {0, 1}},
            "tenure": {"type": "int", "min": 0, "max": 72},
            "MonthlyCharges": {"type": "float", "min": 0.0},
            "TotalCharges": {"type": "float", "min": 0.0},
        }

    def __init__(self, pipeline: Pipeline, allow_unknown_categories: bool = False):
        """Initialize the contract with the production pipeline instance.

        Args:
            pipeline: The EXACT fitted production pipeline from ChurnPredictor.
            allow_unknown_categories: If True, unknown categorical values are
                passed through to the OneHotEncoder (which has handle_unknown="ignore").
                Default False (strict production mode).
        """
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a sklearn Pipeline instance")

        # Lazy import to avoid loading features/config at package import time
        from .features import NUMERIC_COLS, TARGET_COL, _get_categorical_columns

        self._pipeline = pipeline
        self._allow_unknown = allow_unknown_categories

        # Extract preprocessor and model
        self._preprocessor = pipeline.named_steps["preprocessor"]
        self._model = pipeline.named_steps["model"]

        # Derive schema from the fitted pipeline
        self._numeric_features = list(NUMERIC_COLS)
        self._categorical_features = _get_categorical_columns()

        # Extract categorical allowed values from fitted OneHotEncoder
        cat_encoder = self._preprocessor.named_transformers_["cat"].named_steps["encoder"]
        self._categorical_allowed = {
            col: list(cats)
            for col, cats in zip(self._categorical_features, cat_encoder.categories_)
        }

        # All required features in order expected by preprocessor
        self._required_features = self._numeric_features + self._categorical_features

        # Forbidden fields that must not appear in inference input
        self._forbidden_fields = {"customerID", TARGET_COL}

        # All valid feature names (for unexpected field detection)
        self._all_valid_features = set(self._required_features)

    @property
    def numeric_features(self) -> list[str]:
        return self._numeric_features.copy()

    @property
    def categorical_features(self) -> dict[str, dict[str, list[str]]]:
        return {k: {"allowed": v.copy()} for k, v in self._categorical_allowed.items()}

    @property
    def required_features(self) -> list[str]:
        return self._required_features.copy()

    @property
    def forbidden_fields(self) -> set[str]:
        return self._forbidden_fields.copy()

    def validate(self, data: dict[str, Any] | pd.DataFrame) -> ValidationResult:
        """Validate a single inference input.

        Args:
            data: Raw inference input as a dict or single-row DataFrame.
                Must contain all required features, no forbidden fields,
                and values must satisfy type and domain constraints.

        Returns:
            ValidationResult with validated_data as a single-row DataFrame
            with columns in the exact order expected by the preprocessor.

        Raises:
            ValidationError: If any validation check fails, with detailed
                error messages describing all violations.
        """
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            if len(data) != 1:
                raise ValidationError(
                    "validate() expects a single-row input. Use validate_batch() for multiple rows.",
                    ["Input must be a single-row dict or single-row DataFrame"]
                )
            df = data.copy()
        else:
            raise ValidationError(
                "Input must be a dict or single-row DataFrame",
                [f"Received {type(data).__name__}"]
            )

        errors = []

        # Check forbidden fields
        forbidden_present = [col for col in self._forbidden_fields if col in df.columns]
        if forbidden_present:
            errors.append(f"Forbidden fields present: {forbidden_present}")

        # Check missing required fields
        missing = [col for col in self._required_features if col not in df.columns]
        if missing:
            errors.append(f"Missing required fields: {missing}")

        # Check unexpected fields (only if no missing required fields to avoid noise)
        if not missing:
            unexpected = [col for col in df.columns if col not in self._all_valid_features]
            if unexpected:
                errors.append(f"Unexpected fields: {unexpected}")

        if errors:
            raise ValidationError("Input validation failed", errors)

        # Validate the single row as scalar values
        row = df.iloc[0]
        self._validate_numeric_fields(row, errors)
        self._validate_categorical_fields(row, errors)

        if errors:
            raise ValidationError("Input validation failed", errors)

        # Reorder columns to match preprocessor expectation
        validated_df = df[self._required_features].copy()

        return ValidationResult(validated_data=validated_df)

    def validate_batch(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a batch of inference inputs.

        Args:
            df: DataFrame with one row per inference request.

        Returns:
            ValidationResult with validated_data as a DataFrame with columns
            in the exact order expected by the preprocessor.

        Raises:
            ValidationError: If any row fails validation, with errors
                aggregated across all rows.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(
                "validate_batch() expects a DataFrame",
                [f"Received {type(df).__name__}"]
            )

        if df.empty:
            raise ValidationError("Empty DataFrame provided", ["Input DataFrame is empty"])

        errors = []

        # Check forbidden fields
        forbidden_present = [col for col in self._forbidden_fields if col in df.columns]
        if forbidden_present:
            errors.append(f"Forbidden fields present: {forbidden_present}")

        # Check missing required fields
        missing = [col for col in self._required_features if col not in df.columns]
        if missing:
            errors.append(f"Missing required fields: {missing}")

        # Check unexpected fields
        unexpected = [col for col in df.columns if col not in self._all_valid_features]
        if unexpected:
            errors.append(f"Unexpected fields: {unexpected}")

        if errors:
            raise ValidationError("Batch validation failed", errors)

        # Validate each row
        for idx, row in df.iterrows():
            row_errors: list[str] = []
            row_series = pd.Series(row)
            self._validate_numeric_fields(row_series, row_errors, row_idx=idx)
            self._validate_categorical_fields(row_series, row_errors, row_idx=idx)
            errors.extend(row_errors)

        if errors:
            raise ValidationError("Batch validation failed", errors)

        # Reorder columns
        validated_df = df[self._required_features].copy()

        return ValidationResult(validated_data=validated_df)

    def _validate_numeric_fields(self, data: pd.Series | pd.DataFrame, errors: list[str], row_idx: int | None = None):
        """Validate numeric fields for type, finiteness, and constraints."""
        prefix = f"Row {row_idx}: " if row_idx is not None else ""

        for col in self._numeric_features:
            val = data[col]

            # Check for missing/NaN
            if pd.isna(val):
                errors.append(f"{prefix}{col}: missing value")
                continue

            # Type and finiteness check
            constraint = self._default_numeric_constraints().get(col, {})
            expected_type = constraint.get("type", "float")

            try:
                if expected_type == "int":
                    # Reject float representations of integers (e.g., 1.0)
                    if isinstance(val, float) and not val.is_integer():
                        errors.append(f"{prefix}{col}: expected integer, got float {val}")
                        continue
                    val = int(val)
                else:
                    val = float(val)

                if not np.isfinite(val):
                    errors.append(f"{prefix}{col}: non-finite value {val}")
                    continue

                # Constraint checks
                if "allowed" in constraint and val not in constraint["allowed"]:
                    errors.append(f"{prefix}{col}: value {val} not in allowed set {constraint['allowed']}")
                if "min" in constraint and val < constraint["min"]:
                    errors.append(f"{prefix}{col}: value {val} below minimum {constraint['min']}")
                if "max" in constraint and val > constraint["max"]:
                    errors.append(f"{prefix}{col}: value {val} above maximum {constraint['max']}")

            except (ValueError, TypeError):
                errors.append(f"{prefix}{col}: invalid {expected_type} value {val}")

    def _validate_categorical_fields(self, data: pd.Series | pd.DataFrame, errors: list[str], row_idx: int | None = None):
        """Validate categorical fields against fitted encoder categories."""
        prefix = f"Row {row_idx}: " if row_idx is not None else ""

        for col in self._categorical_features:
            val = data[col]

            # Check for missing/NaN
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                errors.append(f"{prefix}{col}: missing value")
                continue

            val_str = str(val).strip()
            allowed = self._categorical_allowed[col]

            if val_str not in allowed:
                if self._allow_unknown:
                    # In lenient mode, we allow it to pass through to OneHotEncoder
                    # which will produce all-zeros for this feature
                    pass
                else:
                    errors.append(
                        f"{prefix}{col}: value '{val_str}' not in allowed categories {allowed}"
                    )


def _extract_categorical_allowed_from_pipeline(pipeline: Pipeline) -> dict[str, list[str]]:
    """Extract categorical allowed values from a fitted pipeline's OneHotEncoder.

    This is a standalone utility for testing and external schema extraction.
    """
    # Lazy import to avoid loading features/config at package import time
    from .features import _get_categorical_columns

    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    return {
        col: list(cats)
        for col, cats in zip(_get_categorical_columns(), cat_encoder.categories_)
    }


def derive_schema_from_pipeline(pipeline: Pipeline) -> dict[str, Any]:
    """Derive the complete data contract schema from a fitted production pipeline.

    Returns a dictionary with the complete schema including:
    - numeric features with constraints
    - categorical features with allowed values
    - forbidden fields
    - feature ordering
    """
    # Lazy import to avoid loading features/config at package import time
    from .features import NUMERIC_COLS, _get_categorical_columns

    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    numeric_features = list(NUMERIC_COLS)
    categorical_features = _get_categorical_columns()

    categorical_allowed = {
        col: list(cats)
        for col, cats in zip(categorical_features, cat_encoder.categories_)
    }

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_allowed,
        "forbidden_fields": ["customerID", "Churn"],
        "feature_order": list(NUMERIC_COLS) + categorical_features,
        "numeric_constraints": DataContract._default_numeric_constraints(),
    }
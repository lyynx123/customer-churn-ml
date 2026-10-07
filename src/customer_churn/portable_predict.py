"""Phase 15 - Portable Inference Runtime for Customer Churn Prediction.

This module provides a completely portable inference runtime that requires only:
- final_pipeline.joblib (fitted sklearn Pipeline)
- inference_schema.json (derived from fitted OneHotEncoder)
- inference_metadata.json (production threshold + integrity hash)

It has NO dependencies on:
- Training data (data/processed/*.parquet)
- features.py, config.py
- Training/evaluation modules
- Notebooks

Public API mirrors ChurnPredictor from Phase 14.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# =============================================================================
# Portable Data Contract - Schema-driven validation (NO features.py dependency)
# =============================================================================

class PortableValidationError(Exception):
    """Raised when input validation fails at the portable inference boundary."""

    def __init__(self, message: str, errors: list[str]):
        self.errors = errors
        full_message = message + ": " + "; ".join(errors) if errors else message
        super().__init__(full_message)


@dataclass(frozen=True)
class PortableValidationResult:
    """Result of successful portable validation."""
    validated_data: pd.DataFrame


class PortableDataContract:
    """Portable data contract for production inference.

    Validates input against a pre-extracted schema JSON artifact.
    Does NOT depend on training code or fitted pipeline for schema discovery.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        allow_unknown_categories: bool = False,
    ):
        """Initialize the contract with a pre-extracted schema.

        Args:
            schema: Dictionary from inference_schema.json
            allow_unknown_categories: If True, unknown categorical values are
                passed through (lenient mode). Default False (strict production).
        """
        self._schema = schema
        self._allow_unknown = allow_unknown_categories

        # Extract from schema
        self._numeric_features = list(schema["numeric_features"])
        self._categorical_features = dict(schema["categorical_features"])
        self._required_features = list(schema["feature_order"])
        self._forbidden_fields = set(schema["forbidden_fields"])
        self._all_valid_features = set(self._required_features)
        self._numeric_constraints = schema["numeric_constraints"]

    @property
    def numeric_features(self) -> list[str]:
        return self._numeric_features.copy()

    @property
    def categorical_features(self) -> dict[str, dict[str, list[str]]]:
        return {k: {"allowed": v.copy()} for k, v in self._categorical_features.items()}

    @property
    def required_features(self) -> list[str]:
        return self._required_features.copy()

    @property
    def forbidden_fields(self) -> set[str]:
        return self._forbidden_fields.copy()

    def validate(self, data: dict[str, Any] | pd.DataFrame) -> PortableValidationResult:
        """Validate a single inference input.

        Args:
            data: Raw inference input as dict or single-row DataFrame.

        Returns:
            PortableValidationResult with validated_data as single-row DataFrame.

        Raises:
            PortableValidationError: If validation fails.
        """
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            if len(data) != 1:
                raise PortableValidationError(
                    "validate() expects a single-row input. Use validate_batch() for multiple rows.",
                    ["Input must be a single-row dict or single-row DataFrame"]
                )
            df = data.copy()
        else:
            raise PortableValidationError(
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

        # Check unexpected fields
        if not missing:
            unexpected = [col for col in df.columns if col not in self._all_valid_features]
            if unexpected:
                errors.append(f"Unexpected fields: {unexpected}")

        if errors:
            raise PortableValidationError("Input validation failed", errors)

        # Validate the single row
        row = df.iloc[0]
        self._validate_numeric_fields(row, errors)
        self._validate_categorical_fields(row, errors)

        if errors:
            raise PortableValidationError("Input validation failed", errors)

        # Reorder columns to match preprocessor expectation
        validated_df = df[self._required_features].copy()

        return PortableValidationResult(validated_data=validated_df)

    def validate_batch(self, df: pd.DataFrame) -> PortableValidationResult:
        """Validate a batch of inference inputs."""
        if not isinstance(df, pd.DataFrame):
            raise PortableValidationError(
                "validate_batch() expects a DataFrame",
                [f"Received {type(df).__name__}"]
            )

        if df.empty:
            raise PortableValidationError("Empty DataFrame provided", ["Input DataFrame is empty"])

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
            raise PortableValidationError("Batch validation failed", errors)

        # Validate each row
        for idx, row in df.iterrows():
            row_errors: list[str] = []
            row_series = pd.Series(row)
            self._validate_numeric_fields(row_series, row_errors, row_idx=idx)
            self._validate_categorical_fields(row_series, row_errors, row_idx=idx)
            errors.extend(row_errors)

        if errors:
            raise PortableValidationError("Batch validation failed", errors)

        # Reorder columns
        validated_df = df[self._required_features].copy()

        return PortableValidationResult(validated_data=validated_df)

    def _validate_numeric_fields(
        self,
        data: pd.Series,
        errors: list[str],
        row_idx: int | None = None
    ) -> None:
        """Validate numeric fields for type, finiteness, and constraints."""
        prefix = f"Row {row_idx}: " if row_idx is not None else ""

        for col in self._numeric_features:
            val = data[col]

            # Check for missing/NaN
            if pd.isna(val):
                errors.append(f"{prefix}{col}: missing value")
                continue

            constraint = self._numeric_constraints.get(col, {})
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

    def _validate_categorical_fields(
        self,
        data: pd.Series,
        errors: list[str],
        row_idx: int | None = None
    ) -> None:
        """Validate categorical fields against fitted encoder categories."""
        prefix = f"Row {row_idx}: " if row_idx is not None else ""

        for col in self._categorical_features:
            val = data[col]

            # Check for missing/NaN
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                errors.append(f"{prefix}{col}: missing value")
                continue

            val_str = str(val).strip()
            allowed = self._categorical_features[col]

            if val_str not in allowed:
                if self._allow_unknown:
                    # In lenient mode, allow through to OneHotEncoder (all-zeros)
                    pass
                else:
                    errors.append(
                        f"{prefix}{col}: value '{val_str}' not in allowed categories {allowed}"
                    )


# =============================================================================
# Portable Inference Metadata
# =============================================================================

@dataclass(frozen=True)
class InferenceMetadata:
    """Immutable container for portable inference metadata."""
    version: str
    model_type: str
    model_version: str
    production_threshold: float
    threshold_metadata: dict[str, Any]
    preprocessing: dict[str, Any]
    artifact_integrity: dict[str, Any]
    runtime: dict[str, Any]

    @classmethod
    def from_json(cls, path: Path) -> InferenceMetadata:
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            version=data["version"],
            model_type=data["model_type"],
            model_version=data["model_version"],
            production_threshold=data["production_threshold"],
            threshold_metadata=data["threshold_metadata"],
            preprocessing=data["preprocessing"],
            artifact_integrity=data["artifact_integrity"],
            runtime=data["runtime"],
        )

    @property
    def threshold(self) -> float:
        return self.production_threshold


# =============================================================================
# Portable Churn Predictor
# =============================================================================

class PortableChurnPredictor:
    """Portable inference wrapper for the churn prediction model.

    Loads all artifacts from an explicit directory. No training-data dependencies.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        contract: PortableDataContract,
        metadata: InferenceMetadata,
    ):
        """Initialize with pre-loaded components.

        Args:
            pipeline: Fitted sklearn Pipeline (preprocessor + model)
            contract: PortableDataContract with pre-extracted schema
            metadata: InferenceMetadata with threshold and integrity info
        """
        self.pipeline = pipeline
        self._contract = contract
        self._metadata = metadata
        self._threshold = metadata.production_threshold

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: Path | str,
        *,
        allow_unknown_categories: bool = False,
        verify_integrity: bool = True,
    ) -> PortableChurnPredictor:
        """Create predictor by loading all artifacts from a directory.

        This is the primary factory method for production use.

        Args:
            artifact_dir: Directory containing:
                - final_pipeline.joblib
                - inference_schema.json
                - inference_metadata.json
            allow_unknown_categories: Pass through to PortableDataContract.
            verify_integrity: If True, verify SHA256 of pipeline matches metadata.

        Returns:
            PortableChurnPredictor ready for inference.

        Raises:
            FileNotFoundError: If required artifacts missing.
            ValueError: If integrity verification fails.
        """
        artifact_dir = Path(artifact_dir)

        # Load artifacts
        pipeline_path = artifact_dir / "final_pipeline.joblib"
        schema_path = artifact_dir / "inference_schema.json"
        metadata_path = artifact_dir / "inference_metadata.json"

        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        pipeline = joblib.load(pipeline_path)

        with open(schema_path, "r") as f:
            schema = json.load(f)

        metadata = InferenceMetadata.from_json(metadata_path)

        # Verify artifact integrity
        if verify_integrity:
            expected_hash = metadata.artifact_integrity["pipeline_sha256"]
            actual_hash = compute_sha256(pipeline_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Artifact integrity check failed: "
                    f"expected SHA256 {expected_hash}, got {actual_hash}"
                )

        contract = PortableDataContract(
            schema,
            allow_unknown_categories=allow_unknown_categories,
        )

        return cls(pipeline=pipeline, contract=contract, metadata=metadata)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def threshold_metadata(self) -> dict:
        return self._metadata.threshold_metadata.copy()

    def _infer_proba(self, X_validated: pd.DataFrame) -> np.ndarray:
        """Private inference on already-validated data."""
        return self.pipeline.predict_proba(X_validated)[:, 1]

    def _infer_predict(self, X_validated: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Private inference for binary predictions on validated data."""
        proba = self._infer_proba(X_validated)
        thresh = threshold if threshold is not None else self.threshold
        return (proba >= thresh).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities for input data.

        Args:
            X: DataFrame with raw customer features.

        Returns:
            Array of churn probabilities (probability of class 1 = churn).

        Raises:
            PortableValidationError: If input fails validation.
        """
        validated = self._contract.validate(X)
        return self._infer_proba(validated.validated_data)

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Predict churn class labels.

        Args:
            X: DataFrame with raw customer features.
            threshold: Override the default decision threshold.

        Returns:
            Binary predictions (1 = churn, 0 = no churn).

        Raises:
            PortableValidationError: If input fails validation.
        """
        validated = self._contract.validate(X)
        thresh = threshold if threshold is not None else self.threshold
        return self._infer_predict(validated.validated_data, threshold=thresh)

    def predict_batch(
        self,
        X: pd.DataFrame,
        threshold: float | None = None,
        return_proba: bool = True,
    ) -> dict[str, Any]:
        """Batch prediction with detailed output.

        Args:
            X: DataFrame with customer features.
            threshold: Override the default decision threshold.
            return_proba: Whether to include probabilities in output.

        Returns:
            Dictionary with predictions, probabilities (optional), and metadata.

        Raises:
            PortableValidationError: If input fails validation.
        """
        validated = self._contract.validate_batch(X)
        thresh = threshold if threshold is not None else self.threshold
        proba = self._infer_proba(validated.validated_data)
        pred = (proba >= thresh).astype(int)

        result = {
            "predictions": pred.tolist(),
            "threshold": thresh,
            "threshold_metadata": self.threshold_metadata,
        }
        if return_proba:
            result["probabilities"] = proba.tolist()

        return result

    def predict_single(
        self,
        customer_data: dict[str, Any],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Predict for a single customer.

        Args:
            customer_data: Dict with customer features.
            threshold: Override decision threshold.

        Returns:
            Dict with prediction details.

        Raises:
            PortableValidationError: If input fails validation.
        """
        validated = self._contract.validate(customer_data)
        thresh = threshold if threshold is not None else self.threshold
        proba = self._infer_proba(validated.validated_data)[0]
        pred = int(proba >= thresh)

        return {
            "churn_probability": float(proba),
            "threshold": thresh,
            "prediction": pred,
            "prediction_label": "Yes" if pred == 1 else "No",
            "threshold_metadata": self.threshold_metadata,
        }


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# CLI Entry Point
# =============================================================================

def _cli() -> int:
    """Minimal CLI for portable batch inference."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Portable Churn Inference CLI"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV/Parquet file with customer features",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("models"),
        help="Artifact directory (default: models)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file for predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (default: 0.28 from metadata)",
    )
    parser.add_argument(
        "--no-integrity-check",
        action="store_true",
        help="Skip SHA256 verification (not recommended)",
    )
    args = parser.parse_args()

    try:
        # Load predictor
        predictor = PortableChurnPredictor.from_artifacts(
            artifact_dir=args.artifacts,
            verify_integrity=not args.no_integrity_check,
        )

        # Load input
        if args.input.suffix == ".parquet":
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)

        # Predict
        result = predictor.predict_batch(df, threshold=args.threshold)

        # Create output DataFrame
        out_df = df.copy()
        out_df["churn_prediction"] = result["predictions"]
        out_df["churn_probability"] = result["probabilities"]

        # Save
        out_df.to_csv(args.output, index=False)

        print(f"✓ Predictions saved to {args.output}")
        print(f"  Rows: {len(out_df)}")
        print(f"  Threshold: {result['threshold']}")
        print(f"  Churn rate: {np.mean(result['predictions']):.2%}")

        return 0

    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError, pd.errors.ParserError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
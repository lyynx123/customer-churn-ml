"""Verify inference artifacts integrity and correctness.

This script verifies that the Phase 15 portable inference artifacts are complete,
valid, and match the expected production configuration.

Usage:
    uv run python scripts/verify_inference_artifacts.py --artifacts-dir models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

REQUIRED_FILES = [
    "final_pipeline.joblib",
    "inference_schema.json",
    "inference_metadata.json",
]

EXPECTED_THRESHOLD = 0.28
EXPECTED_SOURCE_PHASE = "Phase 11"
EXPECTED_SELECTION_METHOD = "Business cost optimization on OOF probabilities"
EXPECTED_NUMERIC_FEATURES = [
    "MonthlyCharges",
    "SeniorCitizen",
    "TotalCharges",
    "tenure",
]
EXPECTED_CATEGORICAL_FEATURES = [
    "Contract",
    "Dependents",
    "DeviceProtection",
    "InternetService",
    "MultipleLines",
    "OnlineBackup",
    "OnlineSecurity",
    "PaperlessBilling",
    "Partner",
    "PaymentMethod",
    "PhoneService",
    "StreamingMovies",
    "StreamingTV",
    "TechSupport",
    "gender",
]
EXPECTED_FEATURE_ORDER = EXPECTED_NUMERIC_FEATURES + EXPECTED_CATEGORICAL_FEATURES
EXPECTED_FORBIDDEN_FIELDS = ["customerID", "Churn"]


class VerificationError(Exception):
    """Raised when artifact verification fails."""


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_files_exist(artifacts_dir: Path) -> None:
    """Verify all required artifact files exist."""
    missing = [f for f in REQUIRED_FILES if not (artifacts_dir / f).exists()]
    if missing:
        raise VerificationError(f"Missing required artifact files: {missing}")


def verify_schema_json(artifacts_dir: Path) -> dict[str, Any]:
    """Verify inference_schema.json is valid and has correct structure."""
    schema_path = artifacts_dir / "inference_schema.json"
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        raise VerificationError(f"inference_schema.json is not valid JSON: {e}")

    # Required top-level keys
    required_keys = [
        "version",
        "numeric_features",
        "categorical_features",
        "feature_order",
        "forbidden_fields",
        "numeric_constraints",
    ]
    for key in required_keys:
        if key not in schema:
            raise VerificationError(f"Schema missing required key: {key}")

    # Verify numeric features
    if set(schema["numeric_features"]) != set(EXPECTED_NUMERIC_FEATURES):
        raise VerificationError(
            f"Schema numeric_features mismatch. "
            f"Expected {EXPECTED_NUMERIC_FEATURES}, got {schema['numeric_features']}"
        )

    # Verify categorical features
    schema_cat_keys = set(schema["categorical_features"].keys())
    expected_cat_keys = set(EXPECTED_CATEGORICAL_FEATURES)
    if schema_cat_keys != expected_cat_keys:
        raise VerificationError(
            f"Schema categorical_features keys mismatch. "
            f"Expected {expected_cat_keys}, got {schema_cat_keys}"
        )

    # Verify each categorical feature has non-empty allowed values
    for col, allowed in schema["categorical_features"].items():
        if not isinstance(allowed, list) or len(allowed) == 0:
            raise VerificationError(
                f"Categorical feature '{col}' has empty or invalid allowed values"
            )

    # Verify feature order
    if schema["feature_order"] != EXPECTED_FEATURE_ORDER:
        raise VerificationError(
            f"Schema feature_order mismatch. "
            f"Expected {EXPECTED_FEATURE_ORDER}, got {schema['feature_order']}"
        )

    # Verify forbidden fields
    if set(schema["forbidden_fields"]) != set(EXPECTED_FORBIDDEN_FIELDS):
        raise VerificationError(
            f"Schema forbidden_fields mismatch. "
            f"Expected {EXPECTED_FORBIDDEN_FIELDS}, got {schema['forbidden_fields']}"
        )

    # Verify numeric constraints exist for all numeric features
    for feat in EXPECTED_NUMERIC_FEATURES:
        if feat not in schema["numeric_constraints"]:
            raise VerificationError(f"Missing numeric_constraints for {feat}")

    return schema


def verify_metadata_json(artifacts_dir: Path) -> dict[str, Any]:
    """Verify inference_metadata.json is valid and has correct production config."""
    metadata_path = artifacts_dir / "inference_metadata.json"
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise VerificationError(f"inference_metadata.json is not valid JSON: {e}")

    # Required top-level keys
    required_keys = [
        "version",
        "model_type",
        "model_version",
        "production_threshold",
        "threshold_metadata",
        "preprocessing",
        "artifact_integrity",
        "runtime",
    ]
    for key in required_keys:
        if key not in metadata:
            raise VerificationError(f"Metadata missing required key: {key}")

    # Verify production threshold
    if metadata["production_threshold"] != EXPECTED_THRESHOLD:
        raise VerificationError(
            f"production_threshold mismatch. "
            f"Expected {EXPECTED_THRESHOLD}, got {metadata['production_threshold']}"
        )

    # Verify threshold_metadata
    tm = metadata["threshold_metadata"]
    if tm.get("value") != EXPECTED_THRESHOLD:
        raise VerificationError(
            f"threshold_metadata.value mismatch. "
            f"Expected {EXPECTED_THRESHOLD}, got {tm.get('value')}"
        )
    if tm.get("source_phase") != EXPECTED_SOURCE_PHASE:
        raise VerificationError(
            f"threshold_metadata.source_phase mismatch. "
            f"Expected '{EXPECTED_SOURCE_PHASE}', got '{tm.get('source_phase')}'"
        )
    if tm.get("selection_method") != EXPECTED_SELECTION_METHOD:
        raise VerificationError(
            f"threshold_metadata.selection_method mismatch. "
            f"Expected '{EXPECTED_SELECTION_METHOD}', got '{tm.get('selection_method')}'"
        )

    # Verify artifact_integrity
    ai = metadata["artifact_integrity"]
    if ai.get("algorithm") != "sha256":
        raise VerificationError(
            f"artifact_integrity.algorithm must be 'sha256', got '{ai.get('algorithm')}'"
        )
    if "pipeline_sha256" not in ai:
        raise VerificationError("artifact_integrity missing pipeline_sha256")

    # Verify preprocessing matches expected
    proc = metadata["preprocessing"]
    if set(proc.get("numeric_features", [])) != set(EXPECTED_NUMERIC_FEATURES):
        raise VerificationError(
            f"preprocessing.numeric_features mismatch. "
            f"Expected {EXPECTED_NUMERIC_FEATURES}, got {proc.get('numeric_features')}"
        )
    if set(proc.get("categorical_features", [])) != set(EXPECTED_CATEGORICAL_FEATURES):
        raise VerificationError(
            f"preprocessing.categorical_features mismatch. "
            f"Expected {EXPECTED_CATEGORICAL_FEATURES}, got {proc.get('categorical_features')}"
        )

    return metadata


def verify_sha256_match(artifacts_dir: Path, metadata: dict[str, Any]) -> None:
    """Verify pipeline SHA256 matches metadata."""
    pipeline_path = artifacts_dir / "final_pipeline.joblib"
    expected_hash = metadata["artifact_integrity"]["pipeline_sha256"]
    actual_hash = compute_sha256(pipeline_path)

    if actual_hash != expected_hash:
        raise VerificationError(
            f"SHA256 mismatch. "
            f"Expected {expected_hash}, got {actual_hash}"
        )


def verify_pipeline_loads(artifacts_dir: Path) -> Pipeline:
    """Verify pipeline loads successfully and has expected structure."""
    pipeline_path = artifacts_dir / "final_pipeline.joblib"
    try:
        pipeline = joblib.load(pipeline_path)
    except (OSError, EOFError, ValueError, TypeError) as e:
        raise VerificationError(f"Failed to load pipeline: {e}")

    if not isinstance(pipeline, Pipeline):
        raise VerificationError(f"Loaded object is not a sklearn Pipeline: {type(pipeline)}")

    # Verify pipeline has expected steps
    if "preprocessor" not in pipeline.named_steps:
        raise VerificationError("Pipeline missing 'preprocessor' step")
    if "model" not in pipeline.named_steps:
        raise VerificationError("Pipeline missing 'model' step")

    # Verify model is RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    model = pipeline.named_steps["model"]
    if not isinstance(model, RandomForestClassifier):
        raise VerificationError(f"Model is not RandomForestClassifier: {type(model)}")

    # Verify model hyperparameters match Phase 8C
    expected_params = {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
    }
    for param, expected_value in expected_params.items():
        actual_value = getattr(model, param, None)
        if actual_value != expected_value:
            raise VerificationError(
                f"Model parameter {param} mismatch. "
                f"Expected {expected_value}, got {actual_value}"
            )

    # Verify preprocessor structure
    from sklearn.compose import ColumnTransformer
    preprocessor = pipeline.named_steps["preprocessor"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise VerificationError(f"Preprocessor is not ColumnTransformer: {type(preprocessor)}")

    # Verify transformers
    transformer_names = [name for name, _, _ in preprocessor.transformers_]
    if "num" not in transformer_names:
        raise VerificationError("Preprocessor missing 'num' transformer")
    if "cat" not in transformer_names:
        raise VerificationError("Preprocessor missing 'cat' transformer")

    return pipeline


def verify_version_compatibility(schema: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Verify schema and metadata versions are compatible."""
    schema_version = schema.get("version")
    metadata_version = metadata.get("version")

    if schema_version != metadata_version:
        raise VerificationError(
            f"Version mismatch: schema version={schema_version}, metadata version={metadata_version}"
        )

    # Both should be "1.0" for Phase 15
    if schema_version != "1.0":
        raise VerificationError(f"Unexpected schema version: {schema_version}")
    if metadata_version != "1.0":
        raise VerificationError(f"Unexpected metadata version: {metadata_version}")


def verify_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    """Run all verification checks.

    Returns:
        Dictionary with verification results.
    """
    results = {
        "files_exist": False,
        "schema_valid": False,
        "metadata_valid": False,
        "sha256_match": False,
        "pipeline_loads": False,
        "version_compatible": False,
        "all_passed": False,
    }

    print(f"Verifying artifacts in: {artifacts_dir}")

    # 1. Files exist
    verify_files_exist(artifacts_dir)
    results["files_exist"] = True
    print("  ✓ All required files exist")

    # 2. Schema valid
    schema = verify_schema_json(artifacts_dir)
    results["schema_valid"] = True
    print("  ✓ Schema JSON valid and structure correct")

    # 3. Metadata valid
    metadata = verify_metadata_json(artifacts_dir)
    results["metadata_valid"] = True
    print("  ✓ Metadata JSON valid and production config correct")

    # 4. SHA256 match
    verify_sha256_match(artifacts_dir, metadata)
    results["sha256_match"] = True
    print("  ✓ SHA256 integrity verified")

    # 5. Pipeline loads
    _ = verify_pipeline_loads(artifacts_dir)
    results["pipeline_loads"] = True
    print("  ✓ Pipeline loads successfully with correct structure")

    # 6. Version compatibility
    verify_version_compatibility(schema, metadata)
    results["version_compatible"] = True
    print("  ✓ Schema and metadata versions compatible (1.0)")

    results["all_passed"] = True
    print("\n✓ All artifact verification checks PASSED")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Phase 15 portable inference artifacts"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing inference artifacts (default: models)",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir

    if not artifacts_dir.exists():
        print(f"Error: Artifacts directory not found: {artifacts_dir}", file=sys.stderr)
        return 1

    try:
        verify_artifacts(artifacts_dir)
        return 0
    except VerificationError as e:
        print(f"Verification FAILED: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"Unexpected error during verification: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
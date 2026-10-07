"""Extract portable inference artifacts from production pipeline.

This script creates two artifact files from the fitted production pipeline:
- inference_schema.json: Feature schema derived from fitted OneHotEncoder
- inference_metadata.json: Production metadata including threshold 0.28

Usage:
    uv run python scripts/extract_inference_artifacts.py
    uv run python scripts/extract_inference_artifacts.py --artifacts-dir models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_numeric_constraints() -> dict[str, dict[str, Any]]:
    """Define numeric constraints matching Phase 14 DataContract validation semantics.

    These constraints match the validation in contract.py:_default_numeric_constraints()
    to ensure portable validation has identical behavior.
    """
    return {
        "SeniorCitizen": {"type": "int", "allowed": [0, 1]},
        "tenure": {"type": "int", "min": 0, "max": 72},
        "MonthlyCharges": {"type": "float", "min": 0.0},
        "TotalCharges": {"type": "float", "min": 0.0},
    }


def extract_schema_from_pipeline(pipeline_path: Path) -> dict[str, Any]:
    """Extract inference schema from fitted production pipeline.

    Args:
        pipeline_path: Path to final_pipeline.joblib

    Returns:
        Schema dictionary with feature definitions.
    """
    pipeline = joblib.load(pipeline_path)

    preprocessor = pipeline.named_steps["preprocessor"]

    # Extract numeric features (from ColumnTransformer num transformer)
    _, _, num_features = preprocessor.transformers_[0]
    numeric_features = list(num_features)

    # Extract categorical features (from ColumnTransformer cat transformer)
    _, _, cat_features = preprocessor.transformers_[1]
    categorical_features = list(cat_features)

    # Extract categorical allowed values from fitted OneHotEncoder
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    categorical_allowed = {
        col: list(cats)
        for col, cats in zip(categorical_features, encoder.categories_)
    }

    # Feature order expected by preprocessor (numeric first, then categorical)
    feature_order = numeric_features + categorical_features

    schema = {
        "version": "1.0",
        "numeric_features": numeric_features,
        "categorical_features": categorical_allowed,
        "feature_order": feature_order,
        "forbidden_fields": ["customerID", "Churn"],
        "numeric_constraints": extract_numeric_constraints(),
    }
    return schema


def load_business_metadata() -> dict[str, Any]:
    """Load production threshold and business metadata from Phase 11 artifact.

    The authoritative source for the production threshold is business_cost_analysis.json
    generated in Phase 11, NOT the stale model_metadata.json.
    """
    models_dir = Path(__file__).resolve().parents[1] / "models"
    business_path = models_dir / "business_cost_analysis.json"

    with open(business_path, "r") as f:
        business_data = json.load(f)

    return {
        "production_threshold": business_data["selected_threshold"],
        "threshold_metadata": {
            "value": business_data["selected_threshold"],
            "selection_method": "Business cost optimization on OOF probabilities",
            "source_phase": "Phase 11",
            "cost_matrix": business_data["cost_assumptions"],
            "frozen_test_cost": 16650,
        },
    }


def extract_metadata(pipeline_path: Path) -> dict[str, Any]:
    """Extract inference metadata from pipeline and business artifacts."""
    pipeline = joblib.load(pipeline_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Model info
    model_type = type(model).__name__

    # Preprocessing info
    numeric_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    categorical_imputer = preprocessor.named_transformers_["cat"].named_steps["imputer"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    # Business metadata
    business_meta = load_business_metadata()

    # SHA256 of pipeline artifact
    pipeline_sha256 = compute_sha256(pipeline_path)

    metadata = {
        "version": "1.0",
        "model_type": model_type,
        "model_version": "1.0",
        "production_threshold": business_meta["production_threshold"],
        "threshold_metadata": business_meta["threshold_metadata"],
        "preprocessing": {
            "numeric_features": list(preprocessor.transformers_[0][2]),
            "categorical_features": list(preprocessor.transformers_[1][2]),
            "numeric_imputation": getattr(numeric_imputer, "strategy", "median"),
            "categorical_imputation": getattr(categorical_imputer, "strategy", "most_frequent"),
            "categorical_encoding": f"OneHotEncoder(handle_unknown={encoder.handle_unknown!r}, sparse_output={encoder.sparse_output!r})",
        },
        "artifact_integrity": {
            "algorithm": "sha256",
            "pipeline_sha256": pipeline_sha256,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "joblib_version": joblib.__version__,
        },
    }
    return metadata


def write_json_deterministic(obj: dict[str, Any], path: Path) -> None:
    """Write JSON with deterministic formatting for reproducible artifacts."""
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write with sorted keys, consistent indentation
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")  # Ensure trailing newline


def extract_inference_artifacts(
    artifacts_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract portable inference artifacts from production pipeline.

    Args:
        artifacts_dir: Directory containing final_pipeline.joblib and business_cost_analysis.json
        output_dir: Directory to write inference_schema.json and inference_metadata.json

    Returns:
        Tuple of (schema_dict, metadata_dict)
    """
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).resolve().parents[1] / "models"
    if output_dir is None:
        output_dir = artifacts_dir

    pipeline_path = artifacts_dir / "final_pipeline.joblib"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")

    # Extract schema from fitted pipeline
    schema = extract_schema_from_pipeline(pipeline_path)

    # Extract metadata
    metadata = extract_metadata(pipeline_path)

    # Write artifacts
    schema_path = output_dir / "inference_schema.json"
    metadata_path = output_dir / "inference_metadata.json"

    write_json_deterministic(schema, schema_path)
    write_json_deterministic(metadata, metadata_path)

    print(f"✓ inference_schema.json written to {schema_path}")
    print(f"✓ inference_metadata.json written to {metadata_path}")

    return schema, metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract portable inference artifacts from production pipeline"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory containing final_pipeline.joblib and business_cost_analysis.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write extracted artifacts",
    )
    args = parser.parse_args()

    try:
        extract_inference_artifacts(
            artifacts_dir=args.artifacts_dir,
            output_dir=args.output_dir,
        )
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
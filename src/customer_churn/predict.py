"""Model serialization and inference module for Customer Churn Prediction.

This module provides:
1. Serialization of the final trained pipeline (preprocessing + model)
2. Inference API for making predictions on new customer data
3. Configuration for decision threshold
"""

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .features import TARGET_COL, build_preprocessor, load_data
from .models import build_model_pipeline

# Final model configuration (from Phase 8A/8B)
FINAL_RF_CONFIG = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# Threshold candidates from Phase 8B
THRESHOLD_CANDIDATES = [0.45, 0.50, 0.55, 0.60]

# Default threshold from Phase 8B (OOF F1-optimal candidate)
DEFAULT_THRESHOLD = 0.55
THRESHOLD_METADATA = {
    "value": 0.55,
    "selection_method": "OOF F1-optimal candidate",
    "source_phase": "Phase 8B",
}

# Model artifact paths
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def build_final_pipeline() -> Pipeline:
    """Build the final preprocessing + model pipeline.

    Returns a fitted sklearn Pipeline containing:
    - preprocessor: ColumnTransformer with numeric imputation + categorical one-hot encoding
    - model: RandomForestClassifier with tuned hyperparameters

    The pipeline is NOT fitted - caller must call fit() on training data.
    """
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    preprocessor = build_preprocessor()
    return build_model_pipeline(model, preprocessor)


def train_and_serialize_pipeline(output_path: Path | None = None) -> Pipeline:
    """Train the final pipeline on the full training set and serialize it.

    Args:
        output_path: Path to save the serialized pipeline.
                     Defaults to models/final_pipeline.joblib

    Returns:
        The fitted pipeline.
    """
    if output_path is None:
        output_path = PIPELINE_PATH

    # Load training data
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df["Churn"].replace({"No": 0, "Yes": 1}).astype(int).values

    # Build and fit pipeline
    pipeline = build_final_pipeline()
    print(f"Training final pipeline on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)

    # Ensure models directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize
    joblib.dump(pipeline, output_path)
    print(f"Pipeline saved to {output_path}")

    return pipeline


def load_pipeline(pipeline_path: Path | None = None) -> Pipeline:
    """Load the serialized pipeline.

    Args:
        pipeline_path: Path to the serialized pipeline.
                       Defaults to models/final_pipeline.joblib

    Returns:
        Loaded sklearn Pipeline (preprocessing + model).
    """
    if pipeline_path is None:
        pipeline_path = PIPELINE_PATH

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline not found at {pipeline_path}. "
            "Run train_and_serialize_pipeline() first."
        )

    return joblib.load(pipeline_path)


class ChurnPredictor:
    """Inference wrapper for the churn prediction model.

    Provides a clean API for making predictions on new customer data.
    """

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        pipeline_path: Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """Initialize the predictor.

        Args:
            pipeline: Pre-loaded Pipeline (preprocessing + model).
                      If None, loads from pipeline_path.
            pipeline_path: Path to serialized pipeline. Used if pipeline is None.
            threshold: Decision threshold for positive class (churn).
                       Defaults to 0.55 (OOF F1-optimal from Phase 8B).
        """
        if pipeline is not None:
            self.pipeline = pipeline
        elif pipeline_path is not None:
            self.pipeline = load_pipeline(pipeline_path)
        else:
            self.pipeline = load_pipeline()

        # Initialize threshold metadata BEFORE setting threshold
        self._threshold_metadata = THRESHOLD_METADATA.copy()
        self.threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self._threshold = value
        self._threshold_metadata["value"] = value

    @property
    def threshold_metadata(self) -> dict:
        return self._threshold_metadata.copy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities for input data.

        Args:
            X: DataFrame with raw customer features (same columns as training data).
               Must NOT include 'Churn' or 'customerID' columns.

        Returns:
            Array of churn probabilities (probability of class 1 = churn).
        """
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Predict churn class labels.

        Args:
            X: DataFrame with raw customer features.
            threshold: Override the default decision threshold.

        Returns:
            Binary predictions (1 = churn, 0 = no churn).
        """
        proba = self.predict_proba(X)
        thresh = threshold if threshold is not None else self.threshold
        return (proba >= thresh).astype(int)

    def predict_batch(
        self, X: pd.DataFrame, threshold: float | None = None, return_proba: bool = True
    ) -> dict[str, Any]:
        """Batch prediction with detailed output.

        Args:
            X: DataFrame with customer features.
            threshold: Override decision threshold.
            return_proba: Whether to include probabilities in output.

        Returns:
            Dictionary with predictions, probabilities (optional), and metadata.
        """
        proba = self.predict_proba(X)
        thresh = threshold if threshold is not None else self.threshold
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
        self, customer_data: dict[str, Any], threshold: float | None = None
    ) -> dict[str, Any]:
        """Predict for a single customer.

        Args:
            customer_data: Dict with customer features (same keys as training features).
            threshold: Override decision threshold.

        Returns:
            Dict with prediction details.
        """
        X = pd.DataFrame([customer_data])
        proba = self.predict_proba(X)[0]
        thresh = threshold if threshold is not None else self.threshold
        pred = int(proba >= thresh)

        return {
            "churn_probability": float(proba),
            "threshold": thresh,
            "prediction": pred,
            "prediction_label": "Yes" if pred == 1 else "No",
            "threshold_metadata": self.threshold_metadata,
        }


def create_model_metadata(
    pipeline: Pipeline,
    threshold: float = DEFAULT_THRESHOLD,
    train_samples: int = 3451,
) -> dict[str, Any]:
    """Create model metadata dictionary for serialization.

    Args:
        pipeline: Fitted pipeline.
        threshold: Decision threshold used.
        train_samples: Number of training samples.

    Returns:
        Dictionary with model metadata.
    """
    # Extract model from pipeline
    model = pipeline.named_steps["model"]

    return {
        "model_type": "RandomForestClassifier",
        "model_version": "1.0",
        "model_parameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features,
            "class_weight": model.class_weight,
            "random_state": model.random_state,
            "n_jobs": model.n_jobs,
        },
        "random_state": 42,
        "threshold": {
            "value": threshold,
            "selection_method": "OOF F1-optimal candidate",
            "source_phase": "Phase 8B",
        },
        "training_rows": 3451,
        "test_rows": 2113,
        "feature_pipeline_included": True,
        "preprocessing": {
            "numeric_features": [
                "MonthlyCharges",
                "SeniorCitizen",
                "TotalCharges",
                "tenure",
            ],
            "categorical_encoding": "OneHotEncoder(handle_unknown=ignore)",
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
        },
    }


def save_model_metadata(
    pipeline: Pipeline,
    metadata_path: Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Save model metadata to JSON file.

    Args:
        pipeline: Fitted pipeline.
        metadata_path: Path to save metadata JSON.
        threshold: Decision threshold used.

    Returns:
        The metadata dictionary.
    """
    if metadata_path is None:
        metadata_path = METADATA_PATH

    metadata = create_model_metadata(pipeline, threshold)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model metadata saved to {metadata_path}")
    return metadata


def main():
    """Train final model, serialize pipeline, and save metadata."""
    warnings.filterwarnings("ignore")

    print("=" * 80)
    print("PHASE 9: MODEL SERIALIZATION & INFERENCE PREPARATION")
    print("=" * 80)

    # Train and serialize pipeline
    pipeline = train_and_serialize_pipeline()

    # Save metadata
    _ = save_model_metadata(pipeline)

    print("\n" + "=" * 80)
    print("SERIALIZATION COMPLETE")
    print("=" * 80)
    print(f"Pipeline saved to: {PIPELINE_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Default threshold: {DEFAULT_THRESHOLD} (OOF F1-optimal from Phase 8B)")

    return pipeline


if __name__ == "__main__":
    main()

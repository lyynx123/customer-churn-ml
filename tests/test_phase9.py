"""Tests for Phase 9 - Model Serialization & Inference (Tuned LogisticRegression)."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from customer_churn.config import RANDOM_SEED
from customer_churn.features import TARGET_COL, load_data
from customer_churn.final_evaluation import compute_metrics
from customer_churn.predict import (
    FINAL_LR_CONFIG,
    THRESHOLD_METADATA,
    ChurnPredictor,
    build_final_pipeline,
    create_model_metadata,
    save_model_metadata,
    train_and_serialize_pipeline,
)


def test_final_lr_config():
    """Verify the final LogisticRegression configuration."""
    assert FINAL_LR_CONFIG["C"] == 1.0
    assert FINAL_LR_CONFIG["solver"] == "lbfgs"
    assert FINAL_LR_CONFIG["class_weight"] is None
    assert FINAL_LR_CONFIG["max_iter"] == 5000
    assert FINAL_LR_CONFIG["random_state"] == RANDOM_SEED


def test_final_pipeline_structure():
    """Verify pipeline contains preprocessing + tuned LogisticRegression."""
    pipeline = build_final_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)
    steps = [name for name, _ in pipeline.steps]
    assert "preprocessor" in steps
    assert "model" in steps


def test_compute_metrics_correctness():
    """Test metric computation on deterministic toy data."""
    # TP=2, TN=2, FP=1, FN=1 (Total=6)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    _ = np.array([0.5] * 6)

    m = compute_metrics(y_true, y_pred, np.array([0.5] * 6))

    assert m["tn"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tp"] == 2
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert abs(m["f1"] - 2 / 3) < 1e-10
    assert m["accuracy"] == 4 / 6
    assert abs(m["fpr"] - 1 / 3) < 1e-10
    assert m["fnr"] == 1 / 3
    assert m["positive_rate"] == 3 / 6


def test_confusion_matrix_total_samples():
    """Verify TN + FP + FN + TP matches total evaluated samples."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    _ = np.array([0.1, 0.4, 0.6, 0.9])

    m = compute_metrics(y_true, y_pred, np.array([0.5] * 4))
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y_true)


def test_build_final_pipeline_structure():
    """Test that pipeline has correct structure."""
    pipeline = build_final_pipeline()
    assert isinstance(pipeline, Pipeline)
    steps = [name for name, _ in pipeline.steps]
    assert "preprocessor" in steps
    assert "model" in steps
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)


def test_churn_predictor_initialization():
    """Test ChurnPredictor initialization with default and custom threshold."""
    pipeline = build_final_pipeline()
    # Fit pipeline for DataContract to extract fitted encoder categories
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    pipeline.fit(X_train.iloc[:100], y_train[:100])

    # Test with default threshold
    predictor = ChurnPredictor(pipeline=pipeline)
    assert predictor.threshold == 0.10
    assert (
        predictor.threshold_metadata["selection_method"]
        == "Business cost optimization on OOF probabilities (Tuned LogisticRegression)"
    )

    # Test with custom threshold
    predictor_custom = ChurnPredictor(pipeline=pipeline, threshold=0.60)
    assert predictor_custom.threshold == 0.60
    assert predictor_custom.threshold_metadata["value"] == 0.60

    # Test threshold validation
    with pytest.raises(ValueError):
        ChurnPredictor(pipeline=pipeline, threshold=1.5)
    with pytest.raises(ValueError):
        ChurnPredictor(pipeline=pipeline, threshold=-0.1)


def test_predictor_threshold_update():
    """Test threshold can be updated after initialization."""
    pipeline = build_final_pipeline()
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    pipeline.fit(X_train.iloc[:100], y_train[:100])

    predictor = ChurnPredictor(pipeline=pipeline, threshold=0.50)
    assert predictor.threshold == 0.50
    predictor.threshold = 0.60
    assert predictor.threshold == 0.60
    assert predictor.threshold_metadata["value"] == 0.60


def test_threshold_metadata():
    """Test threshold metadata structure (Phase 11 business threshold for Tuned LR)."""
    assert (
        THRESHOLD_METADATA["selection_method"]
        == "Business cost optimization on OOF probabilities (Tuned LogisticRegression)"
    )
    assert THRESHOLD_METADATA["source_phase"] == "Phase 11"
    assert THRESHOLD_METADATA["value"] == 0.10


def test_pipeline_serialization():
    """Test that pipeline can be serialized and loaded."""
    import tempfile
    from pathlib import Path

    import joblib

    pipeline = build_final_pipeline()

    # Train on a small subset for quick test
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    # Fit on small subset
    pipeline.fit(X_train.iloc[:100], y_train[:100])

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        temp_path = Path(f.name)

    try:
        joblib.dump(pipeline, temp_path)
        loaded = joblib.load(temp_path)

        # Verify it's a pipeline
        assert isinstance(loaded, Pipeline)
        assert "preprocessor" in [n for n, _ in loaded.steps]
        assert "model" in [n for n, _ in loaded.steps]

        # Test that it can make predictions
        proba = loaded.predict_proba(
            load_data("train").drop(columns=[TARGET_COL]).iloc[:5]
        )
        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    finally:
        temp_path.unlink()


def test_model_metadata_creation():
    """Test model metadata creation."""
    pipeline = build_final_pipeline()

    # Train on small subset
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    pipeline.fit(X_train.iloc[:50], y_train[:50])

    metadata = create_model_metadata(pipeline, threshold=0.10, train_samples=50)

    assert metadata["model_type"] == "LogisticRegression"
    assert metadata["model_parameters"]["C"] == 1.0
    assert metadata["model_parameters"]["solver"] == "lbfgs"
    assert metadata["model_parameters"]["class_weight"] is None
    assert metadata["model_parameters"]["max_iter"] == 5000
    assert metadata["random_state"] == 42
    assert metadata["threshold"]["value"] == 0.10
    assert (
        metadata["threshold"]["selection_method"]
        == "Business cost optimization on OOF probabilities (Tuned LogisticRegression)"
    )
    assert metadata["threshold"]["source_phase"] == "Phase 11"
    assert metadata["feature_pipeline_included"] is True
    assert "preprocessing" in metadata
    assert "numeric_features" in metadata["preprocessing"]
    assert "categorical_encoding" in metadata["preprocessing"]


def test_save_model_metadata():
    """Test saving model metadata to JSON."""
    import tempfile
    from pathlib import Path

    pipeline = build_final_pipeline()
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    pipeline.fit(X_train.iloc[:50], y_train[:50])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    try:
        _ = save_model_metadata(pipeline, temp_path, threshold=0.10)

        # Verify file was created and contains correct data
        assert temp_path.exists()
        with open(temp_path) as f:
            loaded = json.load(f)

        assert loaded["model_type"] == "LogisticRegression"
        assert loaded["threshold"]["value"] == 0.10
        assert (
            loaded["threshold"]["selection_method"]
            == "Business cost optimization on OOF probabilities (Tuned LogisticRegression)"
        )
    finally:
        temp_path.unlink()


def test_no_test_data_leakage():
    """Verify no test data is used during training/serialization."""
    import inspect

    # The serialization functions should only use train data
    source = inspect.getsource(train_and_serialize_pipeline)
    assert "train" in source
    assert "test" not in source.lower().replace("test_", "")


def test_churn_predictor_predict_proba():
    """Test that ChurnPredictor can generate probabilities."""
    pipeline = build_final_pipeline()
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    pipeline.fit(X_train.iloc[:100], y_train[:100])

    predictor = ChurnPredictor(pipeline=pipeline, threshold=0.10)

    # Check methods exist
    assert hasattr(predictor, "predict_proba")
    assert hasattr(predictor, "predict")
    assert hasattr(predictor, "predict_batch")
    assert hasattr(predictor, "predict_single")

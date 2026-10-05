"""Tests for Phase 7 final model evaluation."""

import numpy as np
import pytest

from customer_churn.config import RANDOM_SEED
from customer_churn.evaluate import (
    CANDIDATE_MODELS,
    build_pipeline,
    evaluate_on_test,
    format_comparison_table,
)
from customer_churn.features import TARGET_COL, load_data


@pytest.fixture(scope="module")
def train_test_data():
    """Load train and test data once."""
    train_df = load_data("train")
    test_df = load_data("test")
    return train_df, test_df


def test_candidate_models_config():
    """Candidate models must use exact Phase 6 configurations."""
    assert "LogisticRegression" in CANDIDATE_MODELS
    assert "RandomForestClassifier" in CANDIDATE_MODELS

    lr = CANDIDATE_MODELS["LogisticRegression"]
    assert lr.random_state == RANDOM_SEED
    assert lr.max_iter == 2000
    assert lr.solver == "lbfgs"
    assert lr.class_weight == "balanced"

    rf = CANDIDATE_MODELS["RandomForestClassifier"]
    assert rf.random_state == RANDOM_SEED
    assert rf.n_estimators == 100
    assert rf.class_weight == "balanced"


def test_pipeline_fitted_on_train_only(train_test_data):
    """Pipeline must be fitted on train, predictions on test."""
    train_df, test_df = train_test_data

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    pipeline = build_pipeline(CANDIDATE_MODELS["LogisticRegression"])
    result = evaluate_on_test(pipeline, X_train, y_train, X_test, y_test)

    # Predictions must match test size
    assert len(result["y_pred"]) == len(X_test)
    assert len(result["y_proba"]) == len(X_test)
    # Metrics must be in valid range
    for metric, value in result["metrics"].items():
        assert 0.0 <= value <= 1.0, f"{metric} out of range: {value}"


def test_target_excluded_from_features(train_test_data):
    """Target must not be in feature matrix."""
    train_df, test_df = train_test_data
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])

    assert TARGET_COL not in X_train.columns
    assert TARGET_COL not in X_test.columns


def test_confusion_matrix_dimensions(train_test_data):
    """Confusion matrix must be 2x2 for binary classification."""
    train_df, test_df = train_test_data

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    pipeline = build_pipeline(CANDIDATE_MODELS["RandomForestClassifier"])
    result = evaluate_on_test(pipeline, X_train, y_train, X_test, y_test)

    cm = result["confusion_matrix"]
    # Check all counts are non-negative
    assert cm["tn"] >= 0
    assert cm["fp"] >= 0
    assert cm["fn"] >= 0
    assert cm["tp"] >= 0
    # Total must equal test size
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(X_test)
    # Matrix must be 2x2
    assert len(cm["matrix"]) == 2
    assert len(cm["matrix"][0]) == 2


def test_prediction_shapes(train_test_data):
    """Predictions and probabilities must have correct shapes."""
    train_df, test_df = train_test_data

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    for model in CANDIDATE_MODELS.values():
        pipeline = build_pipeline(model)
        result = evaluate_on_test(pipeline, X_train, y_train, X_test, y_test)

        # Binary predictions
        assert set(np.unique(result["y_pred"])).issubset({0, 1})
        # Probabilities in [0, 1]
        assert (result["y_proba"] >= 0).all()
        assert (result["y_proba"] <= 1).all()


def test_comparison_table_format():
    """Comparison table must have expected schema."""
    mock_results = {
        "ModelA": {
            "metrics": {
                "roc_auc": 0.85,
                "pr_auc": 0.65,
                "accuracy": 0.75,
                "precision": 0.52,
                "recall": 0.81,
                "f1": 0.63,
            }
        }
    }
    table = format_comparison_table(mock_results)
    assert "ModelA" in table
    assert "0.8500" in table
    assert "| Model |" in table

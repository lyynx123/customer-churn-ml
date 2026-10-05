"""Tests for Phase 8C final evaluation."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from customer_churn.config import RANDOM_SEED
from customer_churn.features import TARGET_COL, load_data
from customer_churn.final_evaluation import (
    FINAL_RF_CONFIG,
    THRESHOLD_CANDIDATES,
    build_final_pipeline,
    compute_metrics,
    run_final_test_evaluation,
)


def test_final_model_config():
    """Verify the tuned RandomForest configuration."""
    assert FINAL_RF_CONFIG["n_estimators"] == 500
    assert FINAL_RF_CONFIG["max_depth"] == 10
    assert FINAL_RF_CONFIG["min_samples_split"] == 2
    assert FINAL_RF_CONFIG["min_samples_leaf"] == 4
    assert FINAL_RF_CONFIG["max_features"] == "sqrt"
    assert FINAL_RF_CONFIG["class_weight"] == "balanced"
    assert FINAL_RF_CONFIG["random_state"] == RANDOM_SEED


def test_threshold_candidates():
    """Verify exactly four pre-specified thresholds are evaluated."""
    assert list(THRESHOLD_CANDIDATES) == [0.45, 0.50, 0.55, 0.60]


def test_final_pipeline_structure():
    """Verify pipeline contains preprocessing + tuned RandomForest."""
    pipeline = build_final_pipeline()
    assert isinstance(pipeline, Pipeline)
    steps = [name for name, _ in pipeline.steps]
    assert "preprocessor" in steps
    assert "model" in steps
    assert isinstance(pipeline.named_steps["model"], RandomForestClassifier)


def test_compute_metrics_correctness():
    """Test metric computation on deterministic toy data."""
    # TP=2, TN=2, FP=1, FN=1 (Total=6)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.8])

    m = compute_metrics(y_true, y_pred, y_proba)

    assert m["tn"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tp"] == 2
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert m["f1"] == 2 / 3
    assert m["accuracy"] == 4 / 6
    assert m["fpr"] == 1 / 3
    assert m["fnr"] == 1 / 3
    assert m["positive_rate"] == 3 / 6


def test_confusion_matrix_total_samples():
    """Verify TN + FP + FN + TP matches total evaluated samples."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(y_true)


def test_probability_prediction_structure():
    """Verify predict_proba output structure."""
    train_df = load_data("train").iloc[:100]
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    pipeline = build_final_pipeline()
    pipeline.fit(X, y)

    y_proba = pipeline.predict_proba(X)
    assert y_proba.ndim == 2
    assert y_proba.shape[1] == 2
    # Ensure rows sum to 1
    np.testing.assert_allclose(y_proba.sum(axis=1), 1.0)


@pytest.mark.slow
def test_full_evaluation_protocol_no_leakage():
    """Smoke test to ensure evaluation protocol runs and respects isolation."""
    # This just ensures it runs without error and returns expected keys
    # Actual leakage audit is manual but this verifies function signature/returns
    results = run_final_test_evaluation()

    assert "metadata" in results
    assert "model_level_metrics" in results
    assert "threshold_evaluations" in results
    assert len(results["threshold_evaluations"]) == len(THRESHOLD_CANDIDATES)
    assert results["metadata"]["train_samples"] == 3451
    assert results["metadata"]["test_samples"] == 2113

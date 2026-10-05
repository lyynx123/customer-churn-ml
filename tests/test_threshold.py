"""Tests for Phase 8B threshold optimization."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from customer_churn.config import RANDOM_SEED
from customer_churn.features import build_preprocessor, load_data
from customer_churn.models import build_model_pipeline
from customer_churn.threshold import (
    THRESHOLD_GRID,
    TUNED_RF_CONFIG,
    compute_threshold_metrics,
    find_optimal_thresholds,
    format_threshold_table,
    get_oof_predictions,
    run_threshold_analysis,
)


def test_tuned_rf_config():
    """Tuned RF config must match Phase 8A selected parameters."""
    assert TUNED_RF_CONFIG["n_estimators"] == 500
    assert TUNED_RF_CONFIG["max_depth"] == 10
    assert TUNED_RF_CONFIG["min_samples_split"] == 2
    assert TUNED_RF_CONFIG["min_samples_leaf"] == 4
    assert TUNED_RF_CONFIG["max_features"] == "sqrt"
    assert TUNED_RF_CONFIG["class_weight"] == "balanced"
    assert TUNED_RF_CONFIG["random_state"] == RANDOM_SEED


def test_threshold_grid_deterministic():
    """Threshold grid must be deterministic and contain required values."""
    from customer_churn.threshold import THRESHOLD_GRID as grid

    assert grid is THRESHOLD_GRID
    for required in [0.30, 0.40, 0.50, 0.60, 0.70]:
        assert required in THRESHOLD_GRID
    # Sorted ascending
    assert THRESHOLD_GRID == sorted(THRESHOLD_GRID)


def test_oof_predictions_cover_all_data():
    """OOF predictions must cover complete training dataset exactly once."""
    y_true, y_proba = get_oof_predictions()
    train_df = load_data("train")

    assert len(y_true) == train_df.shape[0]
    assert len(y_proba) == train_df.shape[0]
    # No NaN in probabilities
    assert not np.isnan(y_proba).any()
    # Probabilities in valid range
    assert (y_proba >= 0).all()
    assert (y_proba <= 1).all()


def test_cv_uses_stratified_kfold():
    """CV must use StratifiedKFold with correct config."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == RANDOM_SEED


def test_candidate_pipeline_structure():
    """Pipeline must contain preprocessing + tuned RandomForest."""
    model = RandomForestClassifier(**TUNED_RF_CONFIG)
    preprocessor = build_preprocessor()
    pipeline = build_model_pipeline(model, preprocessor)

    assert isinstance(pipeline, Pipeline)
    steps = [name for name, _ in pipeline.steps]
    assert "preprocessor" in steps
    assert "model" in steps
    assert isinstance(pipeline.named_steps["model"], RandomForestClassifier)


def test_compute_threshold_metrics_correct():
    """Threshold metric computation must be correct."""
    # Construct known example
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])

    result = compute_threshold_metrics(y_true, y_proba, threshold=0.5)

    # At threshold 0.5: predicted positive = last 3 samples
    # TP=3, FP=0, FN=0, TN=3
    assert result["tp"] == 3
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 3
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["fpr"] == 0.0
    assert result["fnr"] == 0.0
    assert result["positive_rate"] == 0.5


def test_fpr_fnr_derivation():
    """FPR and FNR must be correctly derived from confusion matrix."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    result = compute_threshold_metrics(y_true, y_proba, threshold=0.45)

    # Predicted positive: last 4 (indices 4-7)
    # TP=4, FP=0, FN=0, TN=4
    assert result["tp"] == 4
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 4
    assert result["fpr"] == 0.0
    assert result["fnr"] == 0.0

    # Now with a threshold that creates both FP and FN
    result2 = compute_threshold_metrics(y_true, y_proba, threshold=0.65)
    # Predicted positive: indices 6,7 (proba >= 0.65)
    # TP=2, FP=0, FN=2, TN=4
    assert result2["tp"] == 2
    assert result2["fn"] == 2
    assert result2["fpr"] == 0.0
    assert result2["fnr"] == 0.5


def test_threshold_results_schema():
    """Threshold analysis results must have consistent schema."""
    results = run_threshold_analysis()

    assert len(results) == len(THRESHOLD_GRID)

    required_keys = {
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "fpr",
        "fnr",
        "positive_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    }

    for r in results:
        assert required_keys.issubset(r.keys())
        # All metrics in valid range
        for metric in [
            "precision",
            "recall",
            "f1",
            "accuracy",
            "fpr",
            "fnr",
            "positive_rate",
        ]:
            assert 0.0 <= r[metric] <= 1.0, f"{metric} out of range at {r['threshold']}"


def test_find_optimal_thresholds():
    """Optimal threshold identification must work."""
    results = run_threshold_analysis()
    optimal = find_optimal_thresholds(results)

    assert "default" in optimal
    assert "f1_optimal" in optimal
    assert optimal["default"] == 0.50
    # F1 optimal must be a valid threshold
    assert optimal["f1_optimal"] in THRESHOLD_GRID


def test_format_threshold_table():
    """Table formatting must include required thresholds."""
    results = run_threshold_analysis()
    table = format_threshold_table(results)

    assert "| Threshold |" in table
    for required in ["0.30", "0.40", "0.50", "0.60", "0.70"]:
        assert required in table

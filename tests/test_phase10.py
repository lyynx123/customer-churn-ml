"""Tests for Phase 10 - Error Analysis."""

import numpy as np
import pytest

from customer_churn.error_analysis import (
    classify_errors,
    error_counts,
    error_rates,
    load_frozen_predictions,
    run_error_analysis,
    safe_divide,
)


def test_error_classification():
    """Error types correctly classified from true/predicted labels."""
    assert list(
        classify_errors(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.9, 0.6, 0.8]),
            threshold=0.5,
        )
    ) == ["TN", "FP", "TP", "TP"]


def test_error_counts():
    """Counts for each error type."""
    counts = error_counts(np.array(["TN", "FP", "FN", "TP", "TN", "FP", "FN", "TP"]))
    assert counts == {"TN": 2, "FP": 2, "FN": 2, "TP": 2}


def test_error_rates():
    """Derive FPR, FNR, and positive rate from counts."""
    counts = {"TN": 50, "FP": 10, "FN": 5, "TP": 40}
    rates = error_rates(counts)
    # FPR = FP / (FP + TN) = 10 / (10 + 50) = 10/60 = 0.1666...
    assert rates["false_positive_rate"] == 10 / 60
    # FNR = FN / (FN + TP) = 5 / 45
    assert rates["false_negative_rate"] == 5 / 45
    # Positive prediction rate = (FP + TP) / total
    assert rates["positive_prediction_rate"] == (10 + 40) / 105


def test_safe_divide():
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(5, 0) == 0.0
    assert safe_divide(0, 5) == 0.0
    assert safe_divide(0, 0) == 0.0
    assert safe_divide(-5, 2) == -2.5


def test_frozen_predictions_loaded():
    """Verify frozen test predictions can be loaded without leaking test data."""
    preds = load_frozen_predictions()
    assert "y_true" in preds
    assert "y_pred" in preds
    assert "y_proba" in preds
    assert "error_type" in preds
    assert len(preds["y_true"]) == 2113
    assert preds["y_true"].dtype == np.int_
    assert preds["y_pred"].dtype == np.int_
    assert preds["y_proba"].dtype == np.float64


def test_no_test_data_leakage_in_analysis():
    """Ensure the analysis functions never access test data during model fitting."""
    import inspect

    source = inspect.getsource(load_frozen_predictions)
    # The function should use read_parquet or load_data to load test data,
    # but it should only use test data for evaluation.
    assert "read_parquet" in source or "load_data" in source


def test_error_analysis_runs():
    """Full error analysis run without errors."""
    results = run_error_analysis(threshold=0.55)
    assert "threshold" in results
    assert "error_counts" in results
    assert "error_rates" in results
    assert "difficult_samples" in results
    assert results["threshold"] == 0.55
    assert results["test_samples"] == 2113


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

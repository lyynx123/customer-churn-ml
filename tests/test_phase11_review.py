"""Phase 11 PR #1 Review Fix Tests."""

import pytest
import numpy as np

from customer_churn.business_cost import (
    CostMatrix,
    build_default_cost_matrix,
    calculate_expected_cost,
    compute_business_cost_metrics,
)
from customer_churn.business_threshold_optimizer import (
    optimize_threshold_business,
    run_business_optimization,
)


def test_default_cost_matrix():
    """Default matrix must have TN=0, FP=10, FN=100, TP=10."""
    costs = build_default_cost_matrix()
    assert costs.cost_tn == 0.0
    assert costs.cost_fp == 10.0
    assert costs.cost_fn == 100.0
    assert costs.cost_tp == 10.0


def test_tp_intervention_cost():
    """Verify TP incurs intervention cost in total cost formula."""
    costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
    # 0 TN, 0 FP, 0 FN, 5 TP
    total_cost = calculate_expected_cost(tn=0, fp=0, fn=0, tp=5, costs=costs)
    assert total_cost == 50.0  # 5 * 10.0


def test_expected_cost_formula():
    """Formula: TN * cost_tn + FP * cost_fp + FN * cost_fn + TP * cost_tp."""
    costs = CostMatrix(cost_tn=1.0, cost_fp=2.0, cost_fn=3.0, cost_tp=4.0)
    total_cost = calculate_expected_cost(tn=10, fp=5, fn=2, tp=8, costs=costs)
    expected = 10 * 1.0 + 5 * 2.0 + 2 * 3.0 + 8 * 4.0
    assert total_cost == expected


def test_threshold_optimizer_deterministic():
    """Optimizer should return exact same results given same inputs."""
    costs = build_default_cost_matrix()
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.2, 0.7, 0.4, 0.6])

    res1 = optimize_threshold_business(y_true, y_proba, costs)
    res2 = optimize_threshold_business(y_true, y_proba, costs)

    assert res1.selected_threshold == res2.selected_threshold
    assert res1.selected_metrics == res2.selected_metrics


def test_optimization_uses_oof_not_test():
    """Ensure run_business_optimization uses OOF and never touches the test set."""
    result = run_business_optimization()
    assert "OOF" in result.optimization_data_source
    assert result.test_set_used_for_optimization is False


def test_optimizer_does_not_read_test_split(monkeypatch):
    """Guard against test-set leakage: optimizer must not load the test split."""
    import customer_churn.threshold as threshold_mod
    import customer_churn.features as features_mod

    forbidden: list[str] = []

    def fake_get_oof_predictions():
        # Synthetic OOF stand-in; spy ensures threshold path stays train-only.
        return (
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([0.1, 0.8, 0.2, 0.7, 0.4, 0.6]),
        )

    original_load = features_mod.load_data

    def spy_load_data(split, *args, **kwargs):
        if split == "test":
            forbidden.append("test")
        return original_load(split, *args, **kwargs)

    monkeypatch.setattr(threshold_mod, "get_oof_predictions", fake_get_oof_predictions)
    # threshold.py binds load_data at import time; patch its namespace too.
    monkeypatch.setattr(features_mod, "load_data", spy_load_data)
    monkeypatch.setattr(threshold_mod, "load_data", spy_load_data)

    result = run_business_optimization()

    assert forbidden == []
    assert 0.0 <= result.selected_threshold <= 1.0

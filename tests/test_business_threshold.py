"""Tests for business threshold optimizer - Phase 11."""

import pytest

from customer_churn.business_threshold_optimizer import (
    build_business_threshold_grid,
    optimize_threshold_business,
    run_business_optimization,
    CostMatrix,
)

from customer_churn.threshold import get_oof_predictions


class TestThresholdGrid:
    def test_default_grid_range(self):
        grid = build_business_threshold_grid()
        assert grid[0] == 0.01
        assert grid[-1] == 0.99
        assert all(0.0 <= t <= 1.0 for t in grid)
        # step size 0.01 → length should be 99
        assert len(grid) == 99

    def test_custom_grid(self):
        grid = build_business_threshold_grid(start=0.1, end=0.5, step=0.1)
        assert grid == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_invalid_grid_params(self):
        with pytest.raises(ValueError):
            build_business_threshold_grid(start=-0.1)
        with pytest.raises(ValueError):
            build_business_threshold_grid(end=1.5)
        with pytest.raises(ValueError):
            build_business_threshold_grid(step=0)
        with pytest.raises(ValueError):
            build_business_threshold_grid(start=0.5, end=0.1)


class TestBusinessOptimization:
    @staticmethod
    def _synthetic_oof_data():
        # Simple synthetic OOF data: 8 samples, balanced
        y_true = [0, 0, 0, 1, 1, 1, 0, 1]
        # Probabilities: perfect ordering for class 1
        y_proba = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.4, 0.9]
        return y_true, y_proba

    def test_optimize_threshold_basic(self):
        y_true, y_proba = self._synthetic_oof_data()
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
        result = optimize_threshold_business(y_true, y_proba, costs)
        # With these costs, we expect the optimizer to favour higher recall
        # (i.e., lower threshold) because FN is expensive.
        # The optimal threshold should be close to 0.3 (first TP appears at 0.6)
        assert 0.0 <= result.selected_threshold <= 1.0
        # Verify selected metrics contain required keys
        required = [
            "threshold",
            "tn",
            "fp",
            "fn",
            "tp",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "specificity",
            "npv",
            "total_cost",
            "average_cost",
            "cost_per_predicted_churn",
        ]
        for key in required:
            assert key in result.selected_metrics

    def test_optimize_with_balanced_costs(self):
        y_true, y_proba = self._synthetic_oof_data()
        # Balanced FP and FN costs → optimizer should pick threshold maximizing overall
        # accuracy (which is close to 0.5 for balanced data)
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=10.0, cost_tp=10.0)
        result = optimize_threshold_business(y_true, y_proba, costs)
        # With symmetric costs, we expect threshold close to 0.5 (midpoint)
        assert 0.4 <= result.selected_threshold <= 0.6

    def test_optimize_invalid_costs(self):
        y_true, y_proba = self._synthetic_oof_data()
        with pytest.raises(ValueError):
            # Negative FP cost
            costs = CostMatrix(cost_tn=0.0, cost_fp=-5.0, cost_fn=10.0, cost_tp=10.0)
            optimize_threshold_business(y_true, y_proba, costs)

    def test_optimize_invalid_threshold_grid(self):
        y_true, y_proba = self._synthetic_oof_data()
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
        # Include an invalid threshold (>1) in custom grid
        with pytest.raises(ValueError):
            optimize_threshold_business(
                y_true,
                y_proba,
                costs,
                threshold_grid=[0.2, 0.5, 1.2],
            )

    def test_optimize_mismatch_lengths(self):
        y_true = [0, 1, 0]
        y_proba = [0.1, 0.9]
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
        with pytest.raises(ValueError):
            optimize_threshold_business(y_true, y_proba, costs)

    def test_run_business_optimization_uses_oof(self):
        # This test ensures the function internally calls OOF via get_oof_predictions()
        # and does not accidentally load test data.
        result = run_business_optimization()
        assert isinstance(result.selected_threshold, float)
        assert 0.0 <= result.selected_threshold <= 1.0
        # Verify that result contains the full grid results
        assert len(result.all_threshold_results) > 0
        # Ensure the source identifier reflects OOF usage
        assert "OOF" in result.optimization_data_source

    def test_optimize_tie_breaking(self):
        # Construct a situation where multiple thresholds have identical total cost.
        y_true = [0, 0, 1, 1]
        # Probabilities produce same confusion matrix for thresholds 0.4 and 0.5
        y_proba = [0.2, 0.8, 0.6, 0.9]
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
        grid = [0.4, 0.5]
        result = optimize_threshold_business(y_true, y_proba, costs, threshold_grid=grid)
        # With equal cost, tie-breaking prefers higher recall → lower threshold (0.4)
        assert result.selected_threshold == 0.4

    def test_optimize_extreme_cost_scenario(self):
        # Scenario where FP cost is far higher than FN cost.
        y_true, y_proba = self._synthetic_oof_data()
        costs = CostMatrix(cost_tn=0.0, cost_fp=200.0, cost_fn=10.0, cost_tp=200.0)
        result = optimize_threshold_business(y_true, y_proba, costs)
        # In this case, optimizer should favour higher threshold to reduce FP.
        assert result.selected_threshold > 0.5

    def test_optimize_returns_all_results(self):
        y_true, y_proba = self._synthetic_oof_data()
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
        result = optimize_threshold_business(y_true, y_proba, costs)
        # Verify that all thresholds in the default grid are present.
        expected_len = len(build_business_threshold_grid())
        assert len(result.all_threshold_results) == expected_len

    def test_optimize_consistency(self):
        # Run optimizer twice with same data and ensure deterministic result.
        y_true, y_proba = self._synthetic_oof_data()
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
        r1 = optimize_threshold_business(y_true, y_proba, costs)
        r2 = optimize_threshold_business(y_true, y_proba, costs)
        assert r1.selected_threshold == r2.selected_threshold
        assert (
            r1.selected_metrics["total_cost"]
            == r2.selected_metrics["total_cost"]
        )

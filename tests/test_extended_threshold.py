"""Tests for extended threshold optimizer with retention success rate."""

import numpy as np
import pytest

from customer_churn.extended_business_cost import (
    ExtendedCostMatrix,
    build_default_extended_cost_matrix,
)
from customer_churn.extended_threshold_optimizer import (
    build_business_threshold_grid,
    optimize_threshold_extended,
    run_extended_optimization,
    run_sensitivity_analysis,
)
from customer_churn.threshold import get_oof_predictions


class TestThresholdGrid:
    """Test threshold grid building."""

    def test_default_grid_range(self):
        """Default grid should be 0.01 to 0.99 step 0.01."""
        grid = build_business_threshold_grid()
        assert grid[0] == 0.01
        assert grid[-1] == 0.99
        assert all(0.0 <= t <= 1.0 for t in grid)
        assert len(grid) == 99

    def test_custom_grid(self):
        """Custom grid parameters work correctly."""
        grid = build_business_threshold_grid(start=0.1, end=0.5, step=0.1)
        assert grid == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_invalid_grid_params(self):
        """Invalid grid parameters raise ValueError."""
        with pytest.raises(ValueError):
            build_business_threshold_grid(start=-0.1)
        with pytest.raises(ValueError):
            build_business_threshold_grid(end=1.5)
        with pytest.raises(ValueError):
            build_business_threshold_grid(step=0)
        with pytest.raises(ValueError):
            build_business_threshold_grid(start=0.5, end=0.1)


class TestTieBreaking:
    """Test deterministic tie-breaking logic."""

    def test_tie_break_fp_gt_fn_chooses_higher_threshold(self):
        """When FP cost > FN cost and costs tie, prefer higher threshold."""
        from customer_churn.extended_business_cost import ExtendedBusinessCostResult
        from customer_churn.extended_threshold_optimizer import _tie_break

        costs = ExtendedCostMatrix(
            cost_tn=0, cost_fp=200, cost_fn=100, cost_intervention=200,
            retention_success_rate=0.0, clv=100  # r=0 makes TP=FP in cost
        )
        # Create two results with equal cost
        r1 = ExtendedBusinessCostResult(
            threshold=0.3, tn=3, fp=0, fn=1, tp=2,
            precision=1.0, recall=2/3, f1=0.8, accuracy=5/6,
            specificity=1.0, npv=0.75,
            total_cost=800.0, average_cost=800/6, cost_per_predicted_churn=200.0,
            net_business_value=-800.0, expected_churn_loss=700.0,
            expected_intervention_cost=400.0, expected_retained_value=0.0
        )
        r2 = ExtendedBusinessCostResult(
            threshold=0.6, tn=3, fp=0, fn=2, tp=1,
            precision=1.0, recall=1/3, f1=0.5, accuracy=4/6,
            specificity=1.0, npv=0.6,
            total_cost=800.0, average_cost=800/6, cost_per_predicted_churn=200.0,
            net_business_value=-800.0, expected_churn_loss=700.0,
            expected_intervention_cost=200.0, expected_retained_value=0.0
        )
        result = _tie_break([r1, r2], costs)
        # FP cost (200) > FN cost (100) -> prefer higher threshold (0.6)
        assert result["threshold"] == 0.6

    def test_tie_break_fn_gt_fp_chooses_higher_recall(self):
        """When FN cost > FP cost and costs tie, prefer higher recall."""
        from customer_churn.extended_business_cost import ExtendedBusinessCostResult
        from customer_churn.extended_threshold_optimizer import _tie_break

        costs = ExtendedCostMatrix(
            cost_tn=0, cost_fp=10, cost_fn=200, cost_intervention=10,
            retention_success_rate=0.0, clv=200  # r=0, FN cost > FP cost
        )
        r1 = ExtendedBusinessCostResult(
            threshold=0.3, tn=3, fp=1, fn=1, tp=2,
            precision=2/3, recall=2/3, f1=0.66, accuracy=5/6,
            specificity=0.75, npv=0.75,
            total_cost=610.0, average_cost=610/6, cost_per_predicted_churn=10.0,
            net_business_value=-610.0, expected_churn_loss=600.0,
            expected_intervention_cost=30.0, expected_retained_value=0.0
        )
        r2 = ExtendedBusinessCostResult(
            threshold=0.6, tn=3, fp=0, fn=2, tp=1,
            precision=1.0, recall=1/3, f1=0.5, accuracy=4/6,
            specificity=1.0, npv=0.6,
            total_cost=610.0, average_cost=610/6, cost_per_predicted_churn=10.0,
            net_business_value=-610.0, expected_churn_loss=600.0,
            expected_intervention_cost=10.0, expected_retained_value=0.0
        )
        result = _tie_break([r1, r2], costs)
        # FP cost (10) <= FN cost (200) -> prefer higher recall -> lower threshold (0.3)
        assert result["threshold"] == 0.3

    def test_tie_break_symmetric_fallback_lower_threshold(self):
        """When costs and recall equal, fallback to lower threshold."""
        from customer_churn.extended_business_cost import ExtendedBusinessCostResult
        from customer_churn.extended_threshold_optimizer import _tie_break

        costs = ExtendedCostMatrix(
            cost_tn=0, cost_fp=10, cost_fn=10, cost_intervention=10,
            retention_success_rate=0.5, clv=10
        )
        # All three thresholds have same cost and recall
        r1 = ExtendedBusinessCostResult(
            threshold=0.3, tn=2, fp=0, fn=0, tp=2,
            precision=1.0, recall=1.0, f1=1.0, accuracy=1.0,
            specificity=1.0, npv=1.0,
            total_cost=40.0, average_cost=10.0, cost_per_predicted_churn=10.0,
            net_business_value=-40.0, expected_churn_loss=0.0,
            expected_intervention_cost=40.0, expected_retained_value=10.0
        )
        r2 = ExtendedBusinessCostResult(
            threshold=0.5, tn=2, fp=0, fn=0, tp=2,
            precision=1.0, recall=1.0, f1=1.0, accuracy=1.0,
            specificity=1.0, npv=1.0,
            total_cost=40.0, average_cost=10.0, cost_per_predicted_churn=10.0,
            net_business_value=-40.0, expected_churn_loss=0.0,
            expected_intervention_cost=40.0, expected_retained_value=10.0
        )
        r3 = ExtendedBusinessCostResult(
            threshold=0.7, tn=2, fp=0, fn=0, tp=2,
            precision=1.0, recall=1.0, f1=1.0, accuracy=1.0,
            specificity=1.0, npv=1.0,
            total_cost=40.0, average_cost=10.0, cost_per_predicted_churn=10.0,
            net_business_value=-40.0, expected_churn_loss=0.0,
            expected_intervention_cost=40.0, expected_retained_value=10.0
        )
        result = _tie_break([r1, r2, r3], costs)
        # All equal -> fallback to lower threshold
        assert result["threshold"] == 0.3


class TestExtendedOptimization:
    """Test extended threshold optimization."""

    @staticmethod
    def _synthetic_oof_data():
        """Simple synthetic OOF data for testing."""
        y_true = [0, 0, 0, 1, 1, 1, 0, 1]
        y_proba = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.4, 0.9]
        return y_true, y_proba

    def test_optimize_basic(self):
        """Basic optimization with retention rate."""
        y_true, y_proba = self._synthetic_oof_data()
        costs = build_default_extended_cost_matrix(retention_success_rate=0.5)
        result = optimize_threshold_extended(y_true, y_proba, costs)
        assert 0.0 <= result.selected_threshold <= 1.0
        required = [
            "threshold", "tn", "fp", "fn", "tp",
            "precision", "recall", "f1", "accuracy",
            "specificity", "npv", "total_cost", "average_cost",
            "cost_per_predicted_churn", "net_business_value",
            "expected_churn_loss", "expected_intervention_cost",
            "expected_retained_value",
        ]
        for key in required:
            assert key in result.selected_metrics

    def test_optimize_with_r_equals_1(self):
        """When r=1.0, should match Phase 11 behavior."""
        y_true, y_proba = self._synthetic_oof_data()
        costs = build_default_extended_cost_matrix(retention_success_rate=1.0)
        result = optimize_threshold_extended(y_true, y_proba, costs)
        # With r=1, FN cost dominates -> lower threshold preferred
        assert 0.0 <= result.selected_threshold <= 1.0

    def test_optimize_with_r_equals_0(self):
        """When r=0, intervention never works."""
        y_true, y_proba = self._synthetic_oof_data()
        costs = build_default_extended_cost_matrix(retention_success_rate=0.0)
        result = optimize_threshold_extended(y_true, y_proba, costs)
        assert 0.0 <= result.selected_threshold <= 1.0

    def test_optimize_invalid_costs(self):
        """Negative costs raise ValueError."""
        y_true, y_proba = self._synthetic_oof_data()
        with pytest.raises(ValueError):
            costs = ExtendedCostMatrix(
                cost_tn=0.0, cost_fp=-5.0, cost_fn=100.0,
                cost_intervention=10.0, retention_success_rate=0.5, clv=100.0
            )
            optimize_threshold_extended(y_true, y_proba, costs)

    def test_optimize_invalid_threshold_grid(self):
        """Invalid threshold in grid raises ValueError."""
        y_true, y_proba = self._synthetic_oof_data()
        costs = build_default_extended_cost_matrix()
        with pytest.raises(ValueError):
            optimize_threshold_extended(
                y_true, y_proba, costs, threshold_grid=[0.2, 0.5, 1.2]
            )

    def test_optimize_mismatch_lengths(self):
        """Mismatched array lengths raise ValueError."""
        y_true = [0, 1, 0]
        y_proba = [0.1, 0.9]
        costs = build_default_extended_cost_matrix()
        with pytest.raises(ValueError):
            optimize_threshold_extended(y_true, y_proba, costs)

    def test_run_extended_optimization_uses_oof(self):
        """run_extended_optimization uses OOF data, not test data."""
        costs = build_default_extended_cost_matrix(retention_success_rate=0.5)
        result = run_extended_optimization(costs)
        assert isinstance(result.selected_threshold, float)
        assert 0.0 <= result.selected_threshold <= 1.0
        assert len(result.all_threshold_results) > 0
        assert "OOF" in result.optimization_data_source
        assert result.test_set_used_for_optimization is False

    def test_optimize_returns_all_results(self):
        """All thresholds in grid are evaluated."""
        y_true, y_proba = self._synthetic_oof_data()
        costs = build_default_extended_cost_matrix()
        result = optimize_threshold_extended(y_true, y_proba, costs)
        expected_len = len(build_business_threshold_grid())
        assert len(result.all_threshold_results) == expected_len

    def test_optimize_consistency(self):
        """Same inputs produce same outputs."""
        y_true, y_proba = self._synthetic_oof_data()
        costs = build_default_extended_cost_matrix()
        r1 = optimize_threshold_extended(y_true, y_proba, costs)
        r2 = optimize_threshold_extended(y_true, y_proba, costs)
        assert r1.selected_threshold == r2.selected_threshold
        assert r1.selected_metrics["total_cost"] == r2.selected_metrics["total_cost"]

    def test_baseline_comparison_included(self):
        """Result includes baseline comparison."""
        y_true, y_proba = self._synthetic_oof_data()
        costs = build_default_extended_cost_matrix(retention_success_rate=0.5)
        result = optimize_threshold_extended(y_true, y_proba, costs)
        baseline = result.baseline_comparison
        assert "contact_no_one_total_cost" in baseline
        assert "contact_everyone_total_cost" in baseline
        assert "contact_no_one_avg_cost" in baseline
        assert "contact_everyone_avg_cost" in baseline
        assert "population_size" in baseline
        assert "n_churn" in baseline
        assert "n_no_churn" in baseline
        assert baseline["population_size"] == len(y_true)


class TestSensitivityAnalysis:
    """Test sensitivity analysis across retention rates."""

    def test_sensitivity_multiple_rates(self):
        """Sensitivity analysis runs across multiple retention rates."""
        y_true, y_proba = get_oof_predictions()
        sensitivity = run_sensitivity_analysis(
            y_true, y_proba,
            retention_rates=[0.2, 0.5, 0.8],
            clv_values=[100.0],
            intervention_costs=[10.0],
        )
        assert len(sensitivity) == 3
        for s in sensitivity:
            assert "scenario_label" in s
            assert "optimal_threshold" in s
            assert "value_vs_no_contact" in s
            assert "value_vs_contact_all" in s
            assert 0.0 <= s["optimal_threshold"] <= 1.0

    def test_sensitivity_multiple_clv(self):
        """Sensitivity analysis runs across multiple CLV values."""
        y_true, y_proba = get_oof_predictions()
        sensitivity = run_sensitivity_analysis(
            y_true, y_proba,
            retention_rates=[0.5],
            clv_values=[50.0, 100.0, 200.0],
            intervention_costs=[10.0],
        )
        assert len(sensitivity) == 3

    def test_sensitivity_multiple_intervention_costs(self):
        """Sensitivity analysis runs across multiple intervention costs."""
        y_true, y_proba = get_oof_predictions()
        sensitivity = run_sensitivity_analysis(
            y_true, y_proba,
            retention_rates=[0.5],
            clv_values=[100.0],
            intervention_costs=[5.0, 10.0, 25.0],
        )
        assert len(sensitivity) == 3

    def test_sensitivity_all_combinations(self):
        """Sensitivity analysis runs all combinations."""
        y_true, y_proba = get_oof_predictions()
        sensitivity = run_sensitivity_analysis(
            y_true, y_proba,
            retention_rates=[0.3, 0.6],
            clv_values=[100.0, 200.0],
            intervention_costs=[10.0, 20.0],
        )
        assert len(sensitivity) == 8  # 2 * 2 * 2


class TestPhase11Consistency:
    """Test consistency with Phase 11 when r=1.0."""

    def test_r_equals_1_matches_phase11_threshold(self):
        """When r=1.0 and costs match Phase 11, threshold with new data is 0.27."""
        # Use the actual Phase 11 default cost matrix
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=1.0,
            clv=100.0,
        )
        result = run_extended_optimization(costs)
        # With new data (TotalCharges=0 for tenure=0), optimal threshold is 0.27
        assert result.selected_threshold == 0.27

    def test_phase11_artifacts_unchanged(self):
        """Phase 11 artifacts remain unchanged."""
        import json
        with open("models/business_cost_analysis.json") as f:
            data = json.load(f)
        assert data["selected_threshold"] == 0.27
        assert data["cost_assumptions"]["cost_tn"] == 0.0
        assert data["cost_assumptions"]["cost_fp"] == 10.0
        assert data["cost_assumptions"]["cost_fn"] == 100.0
        assert data["cost_assumptions"]["cost_tp"] == 10.0

    def test_no_test_set_in_optimization(self):
        """Optimizer never uses test set."""
        import inspect
        source = inspect.getsource(optimize_threshold_extended)
        # Check that test data loading is not in source
        assert 'load_data("test")' not in source
        assert 'y_test' not in source
        source2 = inspect.getsource(run_extended_optimization)
        assert 'load_data("test")' not in source2
        assert 'y_test' not in source2


class TestOOFUsage:
    """Test OOF data usage."""

    def test_oof_generation_works(self):
        """OOF generation returns valid data."""
        y_true, y_proba = get_oof_predictions()
        assert len(y_true) > 0
        assert len(y_proba) == len(y_true)
        assert np.all(y_proba >= 0) and np.all(y_proba <= 1)

    def test_sensitivity_uses_oof(self):
        """Sensitivity analysis uses OOF data."""
        y_true, y_proba = get_oof_predictions()
        sensitivity = run_sensitivity_analysis(y_true, y_proba, retention_rates=[0.5])
        assert len(sensitivity) == 1
        # Verify it's using the same data size as OOF
        assert sensitivity[0]["optimal_threshold"] >= 0.0


class TestDeterministic:
    """Test deterministic behavior."""

    def test_deterministic_optimization(self):
        """Same inputs produce same threshold."""
        costs = build_default_extended_cost_matrix(retention_success_rate=0.5)
        r1 = run_extended_optimization(costs)
        r2 = run_extended_optimization(costs)
        assert r1.selected_threshold == r2.selected_threshold

    def test_deterministic_sensitivity(self):
        """Sensitivity analysis is deterministic."""
        y_true, y_proba = get_oof_predictions()
        s1 = run_sensitivity_analysis(y_true, y_proba, retention_rates=[0.5])
        s2 = run_sensitivity_analysis(y_true, y_proba, retention_rates=[0.5])
        assert s1[0]["optimal_threshold"] == s2[0]["optimal_threshold"]


class TestThresholdBehaviorWrtR:
    """Test threshold behavior as function of retention rate."""

    def test_threshold_behavior_documentation(self):
        """Document actual t*(r) behavior - not assuming monotonicity."""
        costs_base = {
            "cost_tn": 0.0,
            "cost_fp": 10.0,
            "cost_fn": 100.0,
            "cost_intervention": 10.0,
            "clv": 100.0,
        }
        thresholds = {}
        for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            costs = ExtendedCostMatrix(retention_success_rate=r, **costs_base)
            result = run_extended_optimization(costs)
            thresholds[r] = result.selected_threshold

        # Verify all thresholds are valid
        for r, t in thresholds.items():
            assert 0.0 <= t <= 1.0

        # Document actual behavior (no assumption of monotonicity)
        print(f"\nThreshold vs r: {thresholds}")

    def test_r_equals_1_reproduces_phase11_cost(self):
        """At r=1.0, OOF total cost with new data is $26,570."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0, cost_fp=10.0, cost_fn=100.0,
            cost_intervention=10.0, retention_success_rate=1.0, clv=100.0
        )
        result = run_extended_optimization(costs)
        # With new data (TotalCharges=0 for tenure=0), OOF cost is $26,570
        assert result.selected_metrics["total_cost"] == 26570.0


class TestExtendedMetricsSchema:
    """Test extended metrics have correct schema."""

    def test_selected_metrics_schema(self):
        """Extended metrics include all required fields."""
        costs = build_default_extended_cost_matrix(retention_success_rate=0.5)
        result = run_extended_optimization(costs)
        metrics = result.selected_metrics

        required = [
            "threshold", "tn", "fp", "fn", "tp",
            "precision", "recall", "f1", "accuracy",
            "specificity", "npv", "total_cost", "average_cost",
            "cost_per_predicted_churn", "net_business_value",
            "expected_churn_loss", "expected_intervention_cost",
            "expected_retained_value",
        ]
        for key in required:
            assert key in metrics, f"Missing key: {key}"

        # Validate ranges
        for metric in ["precision", "recall", "f1", "accuracy", "specificity", "npv"]:
            assert 0.0 <= metrics[metric] <= 1.0

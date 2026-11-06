"""Tests for extended business cost calculation with retention success rate."""

import numpy as np
import pytest

from customer_churn.extended_business_cost import (
    ExtendedCostMatrix,
    baseline_costs,
    build_default_extended_cost_matrix,
    calculate_extended_expected_cost,
    compute_extended_business_cost_metrics,
    extended_average_cost_per_customer,
    extended_cost_per_predicted_churn,
)


class TestExtendedCostFormula:
    """Test the extended cost formula with retention success rate."""

    def test_tp_cost_decomposition(self):
        """Verify TP cost = cost_intervention + (1-r)*clv."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        # TP=1, others=0
        total = calculate_extended_expected_cost(tn=0, fp=0, fn=0, tp=1, costs=costs)
        expected = 10.0 + (1.0 - 0.5) * 100.0  # 10 + 50 = 60
        assert total == expected

    def test_fp_cost_only_intervention(self):
        """FP cost is just intervention cost (wasted)."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        total = calculate_extended_expected_cost(tn=0, fp=1, fn=0, tp=0, costs=costs)
        assert total == 10.0

    def test_fn_cost_full_clv(self):
        """FN cost is full CLV (no intervention)."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        total = calculate_extended_expected_cost(tn=0, fp=0, fn=1, tp=0, costs=costs)
        assert total == 100.0

    def test_tn_cost_zero(self):
        """TN cost is cost_tn (typically 0)."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        total = calculate_extended_expected_cost(tn=1, fp=0, fn=0, tp=0, costs=costs)
        assert total == 0.0

    def test_r_equals_1_recovers_phase11(self):
        """When r=1.0, cost_intervention=cost_fp, clv=cost_fn, should match Phase 11."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=1.0,
            clv=100.0,
        )
        # Phase 11: TN=0, FP=10, FN=100, TP=10
        # With TN=10, FP=5, FN=3, TP=7:
        # Phase 11: 10*0 + 5*10 + 3*100 + 7*10 = 0 + 50 + 300 + 70 = 420
        # Extended r=1: 10*0 + 5*10 + 3*100 + 7*(10 + 0*100) = 420
        total_extended = calculate_extended_expected_cost(
            tn=10, fp=5, fn=3, tp=7, costs=costs
        )
        assert total_extended == 420.0

    def test_r_equals_0_tp_never_works(self):
        """When r=0, intervention never works; TP cost = intervention + clv."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.0,
            clv=100.0,
        )
        total = calculate_extended_expected_cost(tn=0, fp=0, fn=0, tp=1, costs=costs)
        # TP cost = 10 + (1-0)*100 = 110 (same as FN + intervention)
        assert total == 110.0

    def test_r_boundary_conditions(self):
        """Test r at boundaries 0 and 1."""
        # r=0
        costs_r0 = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.0,
            clv=100.0,
        )
        tp_cost_r0 = calculate_extended_expected_cost(0, 0, 0, 1, costs_r0)
        assert tp_cost_r0 == 110.0  # intervention + full clv

        # r=1
        costs_r1 = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=1.0,
            clv=100.0,
        )
        tp_cost_r1 = calculate_extended_expected_cost(0, 0, 0, 1, costs_r1)
        assert tp_cost_r1 == 10.0  # only intervention

    def test_validation_negative_costs(self):
        """Negative costs must raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ExtendedCostMatrix(
                cost_tn=-1.0,
                cost_fp=10.0,
                cost_fn=100.0,
                cost_intervention=10.0,
                retention_success_rate=0.5,
                clv=100.0,
            )

    def test_validation_r_out_of_bounds_low(self):
        """r < 0 must raise ValueError."""
        with pytest.raises(ValueError, match="retention_success_rate.*must be in"):
            ExtendedCostMatrix(
                cost_tn=0.0,
                cost_fp=10.0,
                cost_fn=100.0,
                cost_intervention=10.0,
                retention_success_rate=-0.1,
                clv=100.0,
            )

    def test_validation_r_out_of_bounds_high(self):
        """r > 1 must raise ValueError."""
        with pytest.raises(ValueError, match="retention_success_rate.*must be in"):
            ExtendedCostMatrix(
                cost_tn=0.0,
                cost_fp=10.0,
                cost_fn=100.0,
                cost_intervention=10.0,
                retention_success_rate=1.1,
                clv=100.0,
            )

    def test_validation_cost_fn_must_equal_clv(self):
        """cost_fn must equal clv; otherwise ValueError."""
        with pytest.raises(ValueError, match="cost_fn.*must equal clv"):
            ExtendedCostMatrix(
                cost_tn=0.0,
                cost_fp=10.0,
                cost_fn=150.0,
                cost_intervention=10.0,
                retention_success_rate=0.5,
                clv=100.0,
            )

    def test_validation_cost_fp_must_equal_cost_intervention(self):
        """cost_fp must equal cost_intervention; otherwise ValueError."""
        with pytest.raises(ValueError, match="cost_fp.*must equal cost_intervention"):
            ExtendedCostMatrix(
                cost_tn=0.0,
                cost_fp=15.0,
                cost_fn=100.0,
                cost_intervention=10.0,
                retention_success_rate=0.5,
                clv=100.0,
            )

    def test_validation_cost_fn_and_clv_both_changed_valid(self):
        """Both cost_fn and clv changed together is valid."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=200.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=200.0,
        )
        assert costs.cost_fn == 200.0
        assert costs.clv == 200.0

    def test_validation_cost_fp_and_intervention_both_changed_valid(self):
        """Both cost_fp and cost_intervention changed together is valid."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=25.0,
            cost_fn=100.0,
            cost_intervention=25.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        assert costs.cost_fp == 25.0
        assert costs.cost_intervention == 25.0


class TestExtendedAverageCost:
    """Test average cost per customer."""

    def test_average_cost_basic(self):
        """Average cost = total_cost / total_customers."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        avg = extended_average_cost_per_customer(
            tn=100, fp=20, fn=15, tp=65, costs=costs
        )
        total_cost = calculate_extended_expected_cost(100, 20, 15, 65, costs)
        expected_avg = total_cost / 200.0
        assert avg == expected_avg

    def test_average_cost_zero_customers(self):
        """Zero customers returns 0."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        avg = extended_average_cost_per_customer(tn=0, fp=0, fn=0, tp=0, costs=costs)
        assert avg == 0.0


class TestExtendedCostPerPredictedChurn:
    """Test cost per predicted churn."""

    def test_intervention_cost_only(self):
        """Only intervention cost counted for predicted churners."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        cpc = extended_cost_per_predicted_churn(fp=20, tp=50, costs=costs)
        assert cpc == (20 * 10.0 + 50 * 10.0) / 70.0

    def test_no_predictions(self):
        """Zero predicted churn returns 0."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        cpc = extended_cost_per_predicted_churn(fp=0, tp=0, costs=costs)
        assert cpc == 0.0


class TestComputeExtendedBusinessCostMetrics:
    """Test full extended business cost metrics computation."""

    def test_perfect_predictions(self):
        """Perfect predictions: only TP intervention costs."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = [0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.9, 0.8]
        result = compute_extended_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=0.5, costs=costs
        )
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["tp"] == 2
        # TP cost = 2 * (10 + 0.5*100) = 2 * 60 = 120
        assert result["total_cost"] == 120.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_worst_predictions(self):
        """Worst predictions: FP intervention + FN full loss."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = [0, 0, 1, 1]
        y_proba = [0.9, 0.8, 0.1, 0.2]
        result = compute_extended_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=0.5, costs=costs
        )
        assert result["tn"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 2
        assert result["tp"] == 0
        # FP: 2*10 = 20, FN: 2*100 = 200, total = 220
        assert result["total_cost"] == 220.0

    def test_threshold_boundary_0(self):
        """Threshold 0 predicts everything positive."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = [0, 0, 1, 1]
        y_proba = [0.5, 0.3, 0.8, 0.9]
        result = compute_extended_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=0.0, costs=costs
        )
        assert result["tn"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 0
        assert result["tp"] == 2
        # FP: 2*10 = 20, TP: 2*60 = 120, total = 140
        assert result["total_cost"] == 140.0

    def test_threshold_boundary_1(self):
        """Threshold 1 predicts nothing positive."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = [0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.8, 0.9]
        result = compute_extended_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=1.0, costs=costs
        )
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 2
        assert result["tp"] == 0
        # FN: 2*100 = 200
        assert result["total_cost"] == 200.0

    def test_cost_components_decomposition(self):
        """Verify cost decomposition fields."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = [0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.9, 0.8]
        result = compute_extended_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=0.5, costs=costs
        )
        assert "expected_churn_loss" in result
        assert "expected_intervention_cost" in result
        assert "expected_retained_value" in result
        assert "net_business_value" in result
        # TN=2, FP=0, FN=0, TP=2
        # intervention_cost = (0+2)*10 = 20
        # churn_loss = 0*100 + 2*(1-0.5)*100 = 100
        # retained_value = 2*0.5*100 = 100
        assert result["expected_intervention_cost"] == 20.0
        assert result["expected_churn_loss"] == 100.0
        assert result["expected_retained_value"] == 100.0
        assert result["net_business_value"] == -120.0
        assert result["total_cost"] == 120.0


class TestBaselineCosts:
    """Test baseline cost calculations."""

    def test_contact_no_one(self):
        """Contact no one: all churners lost, zero intervention."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = np.array([0, 0, 0, 1, 1, 1])  # 3 churners, 3 non-churners
        baseline = baseline_costs(y_true, costs)
        assert baseline["population_size"] == 6
        assert baseline["n_churn"] == 3
        assert baseline["n_no_churn"] == 3
        assert baseline["contact_no_one_total_cost"] == 3 * 100.0  # 300

    def test_contact_everyone(self):
        """Contact everyone: all pay intervention, churners retained with prob r."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = np.array([0, 0, 0, 1, 1, 1])  # 3 churners, 3 non-churners
        baseline = baseline_costs(y_true, costs)
        # 6 * 10 + 3 * (1-0.5) * 100 = 60 + 150 = 210
        assert baseline["contact_everyone_total_cost"] == 210.0

    def test_baselines_same_population(self):
        """Baselines use identical population as model evaluation."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = np.array([0, 0, 1, 1])
        baseline = baseline_costs(y_true, costs)
        assert baseline["population_size"] == 4
        assert baseline["n_churn"] == 2


class TestDefaultExtendedCostMatrix:
    """Test default matrix builder."""

    def test_default_values(self):
        """Default values match Phase 11 when r=1.0."""
        costs = build_default_extended_cost_matrix(retention_success_rate=1.0)
        assert costs.cost_tn == 0.0
        assert costs.cost_fp == 10.0
        assert costs.cost_fn == 100.0
        assert costs.cost_intervention == 10.0
        assert costs.retention_success_rate == 1.0
        assert costs.clv == 100.0

    def test_custom_values(self):
        """Custom values correctly set."""
        costs = build_default_extended_cost_matrix(
            retention_success_rate=0.3,
            clv=200.0,
            cost_intervention=25.0,
        )
        assert costs.retention_success_rate == 0.3
        assert costs.clv == 200.0
        assert costs.cost_intervention == 25.0
        assert costs.cost_fp == 25.0
        assert costs.cost_fn == 200.0

    def test_validation_cost_fn_must_equal_clv(self):
        """cost_fn must equal clv; otherwise ValueError."""
        with pytest.raises(ValueError, match="cost_fn.*must equal clv"):
            ExtendedCostMatrix(
                cost_tn=0.0,
                cost_fp=10.0,
                cost_fn=150.0,
                cost_intervention=10.0,
                retention_success_rate=0.5,
                clv=100.0,
            )

    def test_validation_cost_fp_must_equal_cost_intervention(self):
        """cost_fp must equal cost_intervention; otherwise ValueError."""
        with pytest.raises(ValueError, match="cost_fp.*must equal cost_intervention"):
            ExtendedCostMatrix(
                cost_tn=0.0,
                cost_fp=15.0,
                cost_fn=100.0,
                cost_intervention=10.0,
                retention_success_rate=0.5,
                clv=100.0,
            )

    def test_validation_cost_fn_and_clv_both_changed_valid(self):
        """Both cost_fn and clv changed together is valid."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=200.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=200.0,
        )
        assert costs.cost_fn == 200.0
        assert costs.clv == 200.0

    def test_validation_cost_fp_and_intervention_both_changed_valid(self):
        """Both cost_fp and cost_intervention changed together is valid."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=25.0,
            cost_fn=100.0,
            cost_intervention=25.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        assert costs.cost_fp == 25.0
        assert costs.cost_intervention == 25.0


class TestDeterministic:
    """Test deterministic behavior."""

    def test_deterministic_cost_calculation(self):
        """Same inputs always produce same outputs."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        for _ in range(10):
            total = calculate_extended_expected_cost(10, 5, 3, 7, costs)
            # At r=0.5: TN=10*0 + FP=5*10 + FN=3*100 + TP=7*(10+0.5*100)
            # = 0 + 50 + 300 + 7*60 = 350 + 420 = 770
            assert total == 770.0

    def test_deterministic_metrics(self):
        """compute_extended_business_cost_metrics is deterministic."""
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
        )
        y_true = [0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.7, 0.8]
        r1 = compute_extended_business_cost_metrics(y_true, y_proba, 0.5, costs)
        r2 = compute_extended_business_cost_metrics(y_true, y_proba, 0.5, costs)
        assert r1 == r2

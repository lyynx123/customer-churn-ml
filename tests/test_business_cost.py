"""Tests for business cost calculation - Phase 11."""

import pytest

from customer_churn.business_cost import (
    CostMatrix,
    average_cost_per_customer,
    calculate_expected_cost,
    compute_business_cost_metrics,
    cost_per_predicted_churn,
    validate_costs,
)


class TestCostFormula:
    """Test the core cost formula: TN*cost_tn + FP*cost_fp + FN*cost_fn + TP*cost_tp."""

    def test_basic_cost_formula(self):
        """Verify total cost = TN*cost_tn + FP*cost_fp + FN*cost_fn + TP*cost_tp."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=5.0, cost_fn=50.0, cost_tp=0.0)
        # Confusion matrix: TN=3, FP=2, FN=1, TP=4
        total = calculate_expected_cost(tn=3, fp=2, fn=1, tp=4, costs=costs)
        expected = 3 * 0.0 + 2 * 5.0 + 1 * 50.0 + 4 * 0.0
        assert total == expected

    def test_all_positive_costs(self):
        """All four confusion-matrix outcomes have non-zero costs."""
        costs = CostMatrix(cost_tn=1.0, cost_fp=2.0, cost_fn=10.0, cost_tp=3.0)
        total = calculate_expected_cost(tn=10, fp=5, fn=3, tp=7, costs=costs)
        expected = 10 * 1.0 + 5 * 2.0 + 3 * 10.0 + 7 * 3.0
        assert total == expected

    def test_zero_costs(self):
        """When all costs are zero, total cost is zero."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=0.0, cost_fn=0.0, cost_tp=0.0)
        total = calculate_expected_cost(tn=100, fp=10, fn=5, tp=50, costs=costs)
        assert total == 0.0

    def test_asymmetric_costs(self):
        """FN much more expensive than FP (typical churn scenario)."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        # Equal confusion counts
        total_equal = calculate_expected_cost(tn=5, fp=5, fn=5, tp=5, costs=costs)
        # More FNs → higher cost
        total_fn_heavy = calculate_expected_cost(tn=5, fp=2, fn=8, tp=5, costs=costs)
        assert total_fn_heavy > total_equal

    def test_cost_validation_negative(self):
        """Negative costs must raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_costs(CostMatrix(cost_tn=-1.0, cost_fp=0.0, cost_fn=0.0, cost_tp=0.0))

        with pytest.raises(ValueError, match="must be non-negative"):
            validate_costs(CostMatrix(cost_tn=0.0, cost_fp=-5.0, cost_fn=0.0, cost_tp=0.0))

        with pytest.raises(ValueError, match="must be non-negative"):
            validate_costs(CostMatrix(cost_tn=0.0, cost_fp=0.0, cost_fn=-10.0, cost_tp=0.0))

        with pytest.raises(ValueError, match="must be non-negative"):
            validate_costs(CostMatrix(cost_tn=0.0, cost_fp=0.0, cost_fn=0.0, cost_tp=-3.0))

    def test_confusion_matrix_total_matches(self):
        """TN + FP + FN + TP must equal total samples."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        total = calculate_expected_cost(tn=100, fp=20, fn=15, tp=65, costs=costs)
        assert total == 20 * 10.0 + 15 * 100.0

    def test_large_numbers(self):
        """Large confusion matrix counts should not overflow."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        total = calculate_expected_cost(
            tn=10000, fp=2000, fn=1500, tp=6500, costs=costs
        )
        assert total == 2000 * 10.0 + 1500 * 100.0


class TestAverageCostPerCustomer:
    """Test average cost per customer calculation."""

    def test_average_cost_basic(self):
        """Average cost = total_cost / total_customers."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        avg = average_cost_per_customer(tn=100, fp=20, fn=15, tp=65, costs=costs)
        total_cost = 20 * 10.0 + 15 * 100.0
        expected_avg = total_cost / 200.0
        assert avg == expected_avg

    def test_average_cost_zero_customers(self):
        """When there are no customers, average cost is 0."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        avg = average_cost_per_customer(tn=0, fp=0, fn=0, tp=0, costs=costs)
        assert avg == 0.0

    def test_average_cost_all_correct(self):
        """If all predictions are correct, average cost depends on cost_tn."""
        costs = CostMatrix(cost_tn=5.0, cost_fp=0.0, cost_fn=0.0, cost_tp=0.0)
        avg = average_cost_per_customer(tn=100, fp=0, fn=0, tp=0, costs=costs)
        assert avg == 5.0


class TestCostPerPredictedChurn:
    """Test cost per predicted churn calculation."""

    def test_cost_per_predicted_churn_basic(self):
        """Only FP contributes to cost of predicted churn."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        cpc = cost_per_predicted_churn(fp=20, tp=50, costs=costs)
        assert cpc == 20 * 10.0 / 70.0

    def test_cost_per_predicted_churn_no_predictions(self):
        """When no one is predicted as churn, return 0."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        cpc = cost_per_predicted_churn(fp=0, tp=0, costs=costs)
        assert cpc == 0.0


class TestComputeBusinessCostMetrics:
    """Test full business cost metrics computation from y_true/y_proba."""

    def test_perfect_predictions(self):
        """Perfect predictions yield minimum cost (intervention cost for TPs)."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
        y_true = [0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.9, 0.8]
        result = compute_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=0.5, costs=costs
        )
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["tp"] == 2
        assert result["total_cost"] == 2 * 10.0  # Only TP intervention costs
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_worst_predictions(self):
        """Worst predictions yield maximum cost."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=10.0)
        y_true = [0, 0, 1, 1]
        y_proba = [0.9, 0.8, 0.1, 0.2]
        result = compute_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=0.5, costs=costs
        )
        assert result["tn"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 2
        assert result["tp"] == 0
        assert result["total_cost"] == 2 * 10.0 + 2 * 100.0

    def test_threshold_boundary_0(self):
        """Threshold at 0 predicts everything positive."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        y_true = [0, 0, 1, 1]
        y_proba = [0.5, 0.3, 0.8, 0.9]
        result = compute_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=0.0, costs=costs
        )
        assert result["tn"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 0
        assert result["tp"] == 2

    def test_threshold_boundary_1(self):
        """Threshold at 1 predicts nothing positive (probabilities < 1)."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        y_true = [0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.8, 0.9]
        result = compute_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=1.0, costs=costs
        )
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 2
        assert result["tp"] == 0

    def test_invalid_threshold_low(self):
        """Threshold below 0 must raise ValueError."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            compute_business_cost_metrics(
                y_true=[0, 1], y_proba=[0.5, 0.5], threshold=-0.1, costs=costs
            )

    def test_invalid_threshold_high(self):
        """Threshold above 1 must raise ValueError."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            compute_business_cost_metrics(
                y_true=[0, 1], y_proba=[0.5, 0.5], threshold=1.5, costs=costs
            )

    def test_length_mismatch(self):
        """Mismatched array lengths must raise ValueError."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_business_cost_metrics(
                y_true=[0, 1], y_proba=[0.5], threshold=0.5, costs=costs
            )

    def test_empty_arrays(self):
        """Empty arrays must raise ValueError."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        with pytest.raises(ValueError, match="Cannot compute metrics on empty"):
            compute_business_cost_metrics(
                y_true=[], y_proba=[], threshold=0.5, costs=costs
            )

    def test_probability_out_of_range(self):
        """Probabilities outside [0, 1] must raise ValueError."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        with pytest.raises(ValueError, match="Probabilities must be in"):
            compute_business_cost_metrics(
                y_true=[0, 1], y_proba=[-0.1, 0.5], threshold=0.5, costs=costs
            )

    def test_probability_out_of_range_high(self):
        """Probabilities above 1 must raise ValueError."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        with pytest.raises(ValueError, match="Probabilities must be in"):
            compute_business_cost_metrics(
                y_true=[0, 1], y_proba=[0.5, 1.5], threshold=0.5, costs=costs
            )

    def test_invalid_labels(self):
        """y_true with labels other than 0/1 must raise ValueError."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        with pytest.raises(ValueError, match="y_true contains labels other than"):
            compute_business_cost_metrics(
                y_true=[0, 2], y_proba=[0.5, 0.5], threshold=0.5, costs=costs
            )

    def test_specificity_and_npv(self):
        """Verify specificity and NPV calculations."""
        costs = CostMatrix(cost_tn=0.0, cost_fp=10.0, cost_fn=100.0, cost_tp=0.0)
        y_true = [0, 0, 0, 1, 1, 1]
        y_proba = [0.1, 0.2, 0.6, 0.4, 0.7, 0.8]
        result = compute_business_cost_metrics(
            y_true=y_true, y_proba=y_proba, threshold=0.5, costs=costs
        )
        # Predicted positive: indices 2,4,5 (proba >= 0.5)
        # TN: indices 0,1 → tn=2
        # FP: index 2 → fp=1
        # FN: index 3 → fn=1
        # TP: indices 4,5 → tp=2
        assert result["specificity"] == 2 / 3
        assert result["npv"] == 2 / 3

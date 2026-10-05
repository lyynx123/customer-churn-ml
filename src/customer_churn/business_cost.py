"""Business Cost Optimization Module - Phase 11.

This module implements a business cost decision layer that selects
an operational probability threshold minimizing expected business cost.

IMPORTANT: This is a decision optimization layer, NOT a model tuning layer.
The underlying RandomForestClassifier and its hyperparameters remain unchanged.
Threshold optimization uses OOF (out-of-fold) predictions from training data
only. The frozen test set is NEVER used for threshold selection.
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class BusinessCostConfig(TypedDict):
    """Configurable business cost matrix.

    Each entry represents the per-observation cost of a particular
    confusion-matrix outcome. Costs should be expressed in the same
    monetary unit (e.g., dollars) for all four cells.

    Illustrative scenario assumptions (NOT real business facts):
        - TN: No churn predicted, customer stays. Cost = 0 (no action needed).
        - FP: Churn predicted but customer stays. Cost = retention campaign
              outreach expense (customer receives an unnecessary retention
              offer, e.g., a discount or incentive).
        - TP: Churn predicted and customer churns but is targeted for
              intervention. Cost = retention campaign outreach expense.
        - FN: No churn predicted but customer churns. Cost = lost customer
              lifetime value (customer leaves without being retained).

    The FN/FP ratio is the key driver of threshold placement.
    """

    cost_tn: float
    cost_fp: float
    cost_fn: float
    cost_tp: float


@dataclass(frozen=True)
class CostMatrix:
    """Immutable, validated business cost matrix.

    Frozen dataclass ensures immutability and hashability for
    reproducible cost sensitivity analysis.
    """

    cost_tn: float
    cost_fp: float
    cost_fn: float
    cost_tp: float
    label: str = "illustrative default"


def validate_costs(costs: CostMatrix) -> None:
    """Validate that all cost values are non-negative.

    Args:
        costs: CostMatrix instance to validate.

    Raises:
        ValueError: If any cost value is negative.
    """
    for name, value in [
        ("cost_tn", costs.cost_tn),
        ("cost_fp", costs.cost_fp),
        ("cost_fn", costs.cost_fn),
        ("cost_tp", costs.cost_tp),
    ]:
        if value < 0:
            raise ValueError(
                f"{name}={value} must be non-negative. "
                "Business costs cannot be negative."
            )


@dataclass
class ConfusionMatrixCounts:
    """Confusion matrix counts for a binary classification."""

    tn: int
    fp: int
    fn: int
    tp: int

    def total(self) -> int:
        """Total number of observations."""
        return self.tn + self.fp + self.fn + self.tp

    def validate(self) -> None:
        """Validate counts are non-negative and consistent."""
        for name, value in [
            ("tn", self.tn),
            ("fp", self.fp),
            ("fn", self.fn),
            ("tp", self.tp),
        ]:
            if value < 0:
                raise ValueError(
                    f"Confusion matrix count {name}={value} must be non-negative."
                )


def calculate_expected_cost(
    tn: int, fp: int, fn: int, tp: int, costs: CostMatrix
) -> float:
    """Calculate total expected business cost from confusion matrix counts.

    Formula:
        total_cost = TN * cost_tn + FP * cost_fp + FN * cost_fn + TP * cost_tp

    Args:
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        tp: True positives.
        costs: Business cost matrix.

    Returns:
        Total expected business cost (same unit as input costs).

    Raises:
        ValueError: If any count is negative or costs are negative.
    """
    validate_costs(costs)

    counts = ConfusionMatrixCounts(tn=tn, fp=fp, fn=fn, tp=tp)
    counts.validate()

    total_cost = (
        tn * costs.cost_tn
        + fp * costs.cost_fp
        + fn * costs.cost_fn
        + tp * costs.cost_tp
    )
    return float(total_cost)


def average_cost_per_customer(
    tn: int, fp: int, fn: int, tp: int, costs: CostMatrix
) -> float:
    """Calculate average cost per customer.

    Args:
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        tp: True positives.
        costs: Business cost matrix.

    Returns:
        Average cost per customer (total_cost / total_customers).
        Returns 0.0 if there are no customers.
    """
    total = tn + fp + fn + tp
    if total == 0:
        return 0.0
    total_cost = calculate_expected_cost(tn, fp, fn, tp, costs)
    return total_cost / total


def cost_per_predicted_churn(
    fp: int, tp: int, costs: CostMatrix
) -> float:
    """Calculate cost per predicted churn.

    Useful to understand marginal cost of intervention flagging.

    Args:
        fp: False positives (predicted churn, actually stay).
        tp: True positives (predicted churn, actually churned).
        costs: Business cost matrix.

    Returns:
        Total cost of predictions / number of predicted churners.
        Returns 0.0 if no predictions were made.
    """
    n_predicted_churn = fp + tp
    if n_predicted_churn == 0:
        return 0.0
    # Both FP and TP receive intervention (TP interventions are "correct" but still cost money)
    total_cost = fp * costs.cost_fp + tp * costs.cost_tp
    return total_cost / n_predicted_churn


class BusinessCostResult(TypedDict):
    """Result of business cost evaluation for a single threshold."""

    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    npv: float
    total_cost: float
    average_cost: float
    cost_per_predicted_churn: float


def compute_business_cost_metrics(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    threshold: float,
    costs: CostMatrix,
) -> BusinessCostResult:
    """Compute all business cost metrics for a given threshold.

    Args:
        y_true: True binary labels (0 or 1).
        y_proba: Predicted probabilities for positive class.
        threshold: Decision threshold.
        costs: Business cost matrix.

    Returns:
        Dictionary with confusion matrix counts, classification metrics,
        and business cost metrics.

    Raises:
        ValueError: If threshold is not in [0, 1] or arrays have mismatched
            lengths or probabilities are outside [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Threshold {threshold} must be in [0, 1]. "
            "Business optimization requires valid probability thresholds."
        )

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_proba has {len(y_proba)} elements. "
            "Both arrays must have equal length."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    if not np.all((y_proba >= 0) & (y_proba <= 1)):
        raise ValueError(
            "Probabilities must be in [0, 1]. Found values outside valid range."
        )

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError(
            f"y_true contains labels other than 0/1: {np.unique(y_true)}"
        )

    y_pred = (y_proba >= threshold).astype(int)

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    total_cost = calculate_expected_cost(int(tn), int(fp), int(fn), int(tp), costs)
    avg_cost = average_cost_per_customer(int(tn), int(fp), int(fn), int(tp), costs)
    cost_pred = cost_per_predicted_churn(int(fp), int(tp), costs)

    return BusinessCostResult(
        threshold=float(threshold),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        specificity=specificity,
        npv=npv,
        total_cost=total_cost,
        average_cost=avg_cost,
        cost_per_predicted_churn=cost_pred,
    )


def build_default_cost_matrix() -> CostMatrix:
    """Build default illustrative cost matrix.

    These values are illustrative assumptions only.
    They represent a telecom churn scenario where missing a churner (FN)
    is more costly than a retention intervention (FP/TP).

    Assumption rationale:
        - FP: Cost of retention intervention given to customer who stays.
        - TP: Cost of retention intervention given to customer who churns.
        - Since both receive intervention, cost_fp and cost_tp are equal.
        - FN: Opportunity/loss cost when customer churns without detection.
        - TN: No action, no cost.
        - We use illustrative values: FN=100, FP=10, TP=10, TN=0.
        - This maintains FN/FP ratio of 10:1.

    These are scenario assumptions. Replace with real business values
    before production use.
    """
    return CostMatrix(
        cost_tn=0.0,
        cost_fp=10.0,
        cost_fn=100.0,
        cost_tp=10.0,
        label="illustrative default: FN=10x FP",
    )


DEFAULT_COST_MATRIX = build_default_cost_matrix()

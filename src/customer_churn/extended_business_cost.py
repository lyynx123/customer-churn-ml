"""Extended Business Cost Model with Retention Success Rate - Phase 11 Extension.

This module extends the Phase 11 business cost model by introducing a
retention success rate parameter (r) that represents the probability
an intervention successfully retains a customer who would otherwise churn.

KEY DISTINCTIONS:
- p̂ = model's predicted churn probability (from RandomForest)
- p = true churn probability (latent, unknown)
- r = P(Retained | Intervention, Churn=1) -- EXTERNAL BUSINESS ASSUMPTION

The dataset contains NO intervention history. r, CLV, and intervention
cost are external business assumptions, NOT learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExtendedCostMatrix:
    """Extended business cost matrix with retention success rate.

    All costs expressed in same monetary unit (e.g., dollars).
    retention_success_rate (r) is an EXTERNAL BUSINESS ASSUMPTION.
    The dataset contains no intervention history; r cannot be learned.

    Fields:
        cost_tn: Cost when no action taken and customer stays (typically 0).
        cost_fp: Cost of intervention given to customer who would not churn (wasted).
        cost_fn: Cost when churner not contacted (lost CLV).
        cost_intervention: Cost of making an intervention (applies to FP and TP).
        retention_success_rate: r in [0, 1], probability intervention retains a churner.
        clv: Customer lifetime value (gross value saved if churner retained).
    """

    cost_tn: float
    cost_fp: float
    cost_fn: float
    cost_intervention: float
    retention_success_rate: float
    clv: float
    label: str = "extended: external assumptions"

    def __post_init__(self) -> None:
        """Validate all cost values and retention success rate."""
        for name, value in [
            ("cost_tn", self.cost_tn),
            ("cost_fp", self.cost_fp),
            ("cost_fn", self.cost_fn),
            ("cost_intervention", self.cost_intervention),
            ("clv", self.clv),
        ]:
            if value < 0:
                raise ValueError(f"{name}={value} must be non-negative.")

        r = self.retention_success_rate
        if not 0.0 <= r <= 1.0:
            raise ValueError(
                f"retention_success_rate={r} must be in [0, 1]. "
                "This is an external business assumption, not a learned parameter."
            )


def validate_extended_costs(costs: ExtendedCostMatrix) -> None:
    """Validate ExtendedCostMatrix (called by calculation functions)."""
    # Validation happens in __post_init__, but keep for explicit calls
    _ = costs  # trigger validation via dataclass __post_init__


@dataclass
class ExtendedConfusionMatrixCounts:
    """Confusion matrix counts for extended cost evaluation."""

    tn: int
    fp: int
    fn: int
    tp: int

    def total(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    def validate(self) -> None:
        for name, value in [
            ("tn", self.tn),
            ("fp", self.fp),
            ("fn", self.fn),
            ("tp", self.tp),
        ]:
            if value < 0:
                raise ValueError(f"Confusion matrix count {name}={value} must be non-negative.")


def calculate_extended_expected_cost(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    costs: ExtendedCostMatrix,
) -> float:
    """Calculate total expected business cost with retention success rate.

    Formula per confusion cell:
        TN: cost_tn
        FP: cost_intervention                    (wasted intervention)
        FN: clv                                  (full churn loss)
        TP: cost_intervention + (1 - r) * clv    (intervention + expected residual loss)

    Total Cost = TN*cost_tn + FP*cost_intervention + FN*clv + TP*(cost_intervention + (1-r)*clv)

    Args:
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        tp: True positives.
        costs: ExtendedCostMatrix with retention success rate.

    Returns:
        Total expected business cost (same unit as input costs).
    """
    validate_extended_costs(costs)

    counts = ExtendedConfusionMatrixCounts(tn=tn, fp=fp, fn=fn, tp=tp)
    counts.validate()

    r = costs.retention_success_rate

    total_cost = (
        tn * costs.cost_tn
        + fp * costs.cost_intervention
        + fn * costs.clv
        + tp * (costs.cost_intervention + (1.0 - r) * costs.clv)
    )
    return float(total_cost)


def extended_average_cost_per_customer(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    costs: ExtendedCostMatrix,
) -> float:
    """Calculate average cost per customer with retention success rate."""
    total = tn + fp + fn + tp
    if total == 0:
        return 0.0
    total_cost = calculate_extended_expected_cost(tn, fp, fn, tp, costs)
    return total_cost / total


def extended_cost_per_predicted_churn(
    fp: int,
    tp: int,
    costs: ExtendedCostMatrix,
) -> float:
    """Calculate cost per predicted churn (intervention cost only)."""
    n_predicted_churn = fp + tp
    if n_predicted_churn == 0:
        return 0.0
    # Both FP and TP receive intervention
    total_intervention_cost = fp * costs.cost_intervention + tp * costs.cost_intervention
    return total_intervention_cost / n_predicted_churn


class ExtendedBusinessCostResult(TypedDict):
    """Result of extended business cost evaluation for a single threshold."""

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
    net_business_value: float
    expected_churn_loss: float
    expected_intervention_cost: float
    expected_retained_value: float


def compute_extended_business_cost_metrics(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    threshold: float,
    costs: ExtendedCostMatrix,
) -> ExtendedBusinessCostResult:
    """Compute all extended business cost metrics for a given threshold.

    Args:
        y_true: True binary labels (0 or 1).
        y_proba: Predicted probabilities for positive class.
        threshold: Decision threshold.
        costs: ExtendedCostMatrix with retention success rate.

    Returns:
        Dictionary with confusion matrix counts, classification metrics,
        and extended business cost metrics.

    Raises:
        ValueError: If inputs are invalid.
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
        raise ValueError(f"y_true contains labels other than 0/1: {np.unique(y_true)}")

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

    # Extended cost calculations
    validate_extended_costs(costs)
    r = costs.retention_success_rate

    total_cost = calculate_extended_expected_cost(
        int(tn), int(fp), int(fn), int(tp), costs
    )
    avg_cost = extended_average_cost_per_customer(
        int(tn), int(fp), int(fn), int(tp), costs
    )
    cost_pred = extended_cost_per_predicted_churn(int(fp), int(tp), costs)

    # Net business value (negative cost = value)
    net_business_value = -total_cost

    # Decompose cost components for transparency
    expected_intervention_cost = (fp + tp) * costs.cost_intervention
    expected_churn_loss = fn * costs.clv + tp * (1.0 - r) * costs.clv
    expected_retained_value = tp * r * costs.clv

    return ExtendedBusinessCostResult(
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
        net_business_value=net_business_value,
        expected_churn_loss=expected_churn_loss,
        expected_intervention_cost=expected_intervention_cost,
        expected_retained_value=expected_retained_value,
    )


def baseline_costs(
    y_true: NDArray[np.int_],
    costs: ExtendedCostMatrix,
) -> dict[str, float]:
    """Compute baseline costs for contact-no-one and contact-everyone.

    Uses the SAME population and outcome counts as model evaluation.
    No leakage: baselines computed on the provided y_true only.

    Args:
        y_true: True labels for the evaluation population.
        costs: ExtendedCostMatrix.

    Returns:
        Dictionary with baseline costs.
    """
    validate_extended_costs(costs)
    y_true = np.asarray(y_true).astype(int)

    n_total = len(y_true)
    n_churn = int(np.sum(y_true == 1))
    n_no_churn = n_total - n_churn
    r = costs.retention_success_rate

    # Contact no one: all churners lost, no intervention cost
    cost_no_contact = n_churn * costs.clv

    # Contact everyone: all pay intervention cost, churners retained with prob r
    cost_contact_all = (
        n_total * costs.cost_intervention
        + n_churn * (1.0 - r) * costs.clv
    )

    return {
        "contact_no_one_total_cost": cost_no_contact,
        "contact_no_one_avg_cost": cost_no_contact / n_total if n_total > 0 else 0.0,
        "contact_everyone_total_cost": cost_contact_all,
        "contact_everyone_avg_cost": cost_contact_all / n_total if n_total > 0 else 0.0,
        "population_size": n_total,
        "n_churn": n_churn,
        "n_no_churn": n_no_churn,
    }


def build_default_extended_cost_matrix(
    retention_success_rate: float = 0.5,
    clv: float = 100.0,
    cost_intervention: float = 10.0,
) -> ExtendedCostMatrix:
    """Build default extended cost matrix with specified retention rate.

    Args:
        retention_success_rate: r in [0, 1]. EXTERNAL ASSUMPTION.
        clv: Customer lifetime value. EXTERNAL ASSUMPTION.
        cost_intervention: Cost per intervention. EXTERNAL ASSUMPTION.

    Returns:
        ExtendedCostMatrix with defaults matching Phase 11 when r=1.0.
    """
    # When r=1.0, cost_intervention=10, clv=100:
    # TP cost = 10 + 0*100 = 10 (matches Phase 11 cost_tp)
    # FP cost = 10 (matches Phase 11 cost_fp)
    # FN cost = 100 (matches Phase 11 cost_fn)
    # TN cost = 0 (matches Phase 11 cost_tn)
    return ExtendedCostMatrix(
        cost_tn=0.0,
        cost_fp=cost_intervention,
        cost_fn=clv,
        cost_intervention=cost_intervention,
        retention_success_rate=retention_success_rate,
        clv=clv,
        label=f"extended: r={retention_success_rate}, clv={clv}, int={cost_intervention}",
    )

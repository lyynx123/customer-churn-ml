"""Extended Threshold Optimizer with Retention Success Rate - Phase 11 Extension.

Searches for the probability threshold that minimizes expected business cost
with retention success rate on OOF (out-of-fold) predictions.

LEAKAGE RULE:
    This optimizer ONLY accepts OOF / validation probabilities.
    Frozen test set probabilities MUST NOT be passed to the optimizer.
    The frozen test set is only used AFTER threshold is selected for final
    evaluation.
"""

import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from .extended_business_cost import (
    ExtendedBusinessCostResult,
    ExtendedCostMatrix,
    baseline_costs,
    compute_extended_business_cost_metrics,
    validate_extended_costs,
)
from .threshold import get_oof_predictions

THRESHOLD_STEP: float = 0.01


@dataclass
class ExtendedOptimizationResult:
    """Result of extended business threshold optimization."""

    selected_threshold: float
    selected_metrics: ExtendedBusinessCostResult
    cost_matrix: ExtendedCostMatrix
    baseline_comparison: dict[str, float]
    all_threshold_results: list[ExtendedBusinessCostResult] = field(
        default_factory=list
    )
    optimization_data_source: str = "OOF (cross-validation on training data)"
    test_set_used_for_optimization: bool = False
    tie_breaking_rule: str = field(
        default=(
            "1. minimum total_cost; "
            "2. if tied and FP cost > FN cost: higher threshold; "
            "3. if tied and FP cost <= FN cost: higher recall; "
            "4. lower threshold"
        )
    )


class ExtendedBusinessOptimizationSummary(TypedDict):
    """Machine-readable summary for JSON artifact."""

    objective: str
    selected_threshold: float
    cost_assumptions: dict[str, Any]
    optimization_summary: dict[str, Any]
    business_metrics: dict[str, Any]
    baseline_comparison: dict[str, float]
    model_level_metrics: dict[str, float]
    confusion_matrix: dict[str, int]
    tie_breaking_rule: str
    optimization_data: dict[str, Any]


def build_business_threshold_grid(
    start: float = 0.01,
    end: float = 0.99,
    step: float = THRESHOLD_STEP,
) -> list[float]:
    """Build deterministic threshold grid for optimization."""
    if not 0.0 <= start <= 1.0:
        raise ValueError(f"start={start} must be in [0, 1].")
    if not 0.0 <= end <= 1.0:
        raise ValueError(f"end={end} must be in [0, 1].")
    if step <= 0:
        raise ValueError(f"step={step} must be positive.")
    if start >= end:
        raise ValueError(f"start={start} must be less than end={end}.")

    n_steps = round((end - start) / step) + 1
    grid = [round(start + i * step, 2) for i in range(n_steps)]
    grid = [t for t in grid if 0.0 <= t <= 1.0]
    return sorted(grid)


def _tie_break(
    results: list[ExtendedBusinessCostResult],
    costs: ExtendedCostMatrix,
    tolerance: float = 1e-6,
) -> ExtendedBusinessCostResult:
    """Select optimal result using deterministic tie-breaking."""
    min_cost = min(r["total_cost"] for r in results)
    tied = [r for r in results if abs(r["total_cost"] - min_cost) < tolerance]

    if len(tied) == 1:
        return tied[0]

    # Cost asymmetry check: compare intervention cost vs expected churn loss
    fp_cost = costs.cost_intervention
    fn_cost = costs.clv

    if fp_cost > fn_cost:
        # Prioritize higher threshold to reduce false positives
        best = max(tied, key=lambda r: r["threshold"])
        return best
    else:
        # Prioritize higher recall when FN cost dominates or equal
        max_recall = max(r["recall"] for r in tied)
        tied = [r for r in tied if abs(r["recall"] - max_recall) < tolerance]
        if len(tied) == 1:
            return tied[0]
        # Final fallback: lower threshold
        return min(tied, key=lambda r: r["threshold"])


def optimize_threshold_extended(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    costs: ExtendedCostMatrix,
    threshold_grid: list[float] | None = None,
) -> ExtendedOptimizationResult:
    """Find threshold minimizing expected business cost with retention rate.

    Args:
        y_true: OOF true labels (must be from training-data CV, NOT test).
        y_proba: OOF predicted probabilities (must be from training-data CV,
            NOT test).
        costs: ExtendedCostMatrix with retention success rate.
        threshold_grid: Optional custom threshold grid. Defaults to
            0.01 to 0.99 with 0.01 step.

    Returns:
        ExtendedOptimizationResult with selected threshold and all evaluation results.

    Raises:
        ValueError: If inputs are invalid.
    """
    validate_extended_costs(costs)

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}. "
            "Both arrays must have equal length."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot optimize on empty arrays.")

    if not np.all((y_proba >= 0) & (y_proba <= 1)):
        raise ValueError("Probabilities must be in [0, 1].")

    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must contain only 0/1 labels.")

    if threshold_grid is None:
        threshold_grid = build_business_threshold_grid()

    # Validate all thresholds
    for t in threshold_grid:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"Threshold {t} is not in [0, 1].")

    all_results: list[ExtendedBusinessCostResult] = [
        compute_extended_business_cost_metrics(y_true, y_proba, t, costs)
        for t in threshold_grid
    ]

    selected = _tie_break(all_results, costs)

    # Compute baselines on same population
    baseline = baseline_costs(y_true, costs)

    return ExtendedOptimizationResult(
        selected_threshold=selected["threshold"],
        selected_metrics=selected,
        all_threshold_results=all_results,
        cost_matrix=costs,
        baseline_comparison=baseline,
    )


def run_extended_optimization(
    costs: ExtendedCostMatrix,
) -> ExtendedOptimizationResult:
    """Run full extended business threshold optimization using OOF predictions.

    Uses `threshold.get_oof_predictions()` which generates OOF predictions
    via cross_val_predict on training data only. NO test set involvement.

    Args:
        costs: ExtendedCostMatrix with retention success rate.

    Returns:
        ExtendedOptimizationResult with selected business-optimal threshold.
    """
    y_true_oof, y_proba_oof = get_oof_predictions()

    result = optimize_threshold_extended(y_true_oof, y_proba_oof, costs)
    result.optimization_data_source = (
        "OOF (5-fold StratifiedKFold cross_val_predict on training data)"
    )
    return result


def run_sensitivity_analysis(
    y_true_oof: NDArray[np.int_],
    y_proba_oof: NDArray[np.float64],
    retention_rates: list[float] | None = None,
    clv_values: list[float] | None = None,
    intervention_costs: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Run cost sensitivity analysis across retention rates and cost assumptions.

    For each scenario, finds the business-optimal threshold on OOF data.

    Args:
        y_true_oof: OOF true labels (training data only).
        y_proba_oof: OOF probabilities (training data only).
        retention_rates: List of r values to test. Default: [0.2, 0.4, 0.6, 0.8].
        clv_values: List of CLV values. Default: [100.0].
        intervention_costs: List of intervention costs. Default: [10.0].

    Returns:
        List of scenario results with cost assumptions and optimal thresholds.
    """
    if retention_rates is None:
        retention_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
    if clv_values is None:
        clv_values = [100.0]
    if intervention_costs is None:
        intervention_costs = [10.0]

    results: list[dict[str, Any]] = []

    for r in retention_rates:
        for clv in clv_values:
            for int_cost in intervention_costs:
                costs = ExtendedCostMatrix(
                    cost_tn=0.0,
                    cost_fp=int_cost,
                    cost_fn=clv,
                    cost_intervention=int_cost,
                    retention_success_rate=r,
                    clv=clv,
                    label=f"extended: r={r}, clv={clv}, int={int_cost}",
                )

                opt = optimize_threshold_extended(y_true_oof, y_proba_oof, costs)
                baseline = opt.baseline_comparison
                metrics = opt.selected_metrics

                results.append(
                    {
                        "scenario_label": costs.label,
                        "retention_success_rate": r,
                        "clv": clv,
                        "cost_intervention": int_cost,
                        "cost_tn": costs.cost_tn,
                        "cost_fp": costs.cost_fp,
                        "cost_fn": costs.cost_fn,
                        "optimal_threshold": opt.selected_threshold,
                        "optimal_total_cost": metrics["total_cost"],
                        "optimal_average_cost": metrics["average_cost"],
                        "optimal_precision": metrics["precision"],
                        "optimal_recall": metrics["recall"],
                        "optimal_f1": metrics["f1"],
                        "optimal_accuracy": metrics["accuracy"],
                        "optimal_tn": metrics["tn"],
                        "optimal_fp": metrics["fp"],
                        "optimal_fn": metrics["fn"],
                        "optimal_tp": metrics["tp"],
                        "value_vs_no_contact": baseline["contact_no_one_total_cost"]
                        - metrics["total_cost"],
                        "value_vs_contact_all": baseline["contact_everyone_total_cost"]
                        - metrics["total_cost"],
                        "contact_no_one_cost": baseline["contact_no_one_total_cost"],
                        "contact_everyone_cost": baseline[
                            "contact_everyone_total_cost"
                        ],
                    }
                )

    return results


def build_extended_optimization_summary(
    result: ExtendedOptimizationResult,
    model_level_metrics: dict[str, float] | None = None,
) -> ExtendedBusinessOptimizationSummary:
    """Build machine-readable summary for JSON artifact."""
    metrics = result.selected_metrics
    costs = result.cost_matrix
    baseline = result.baseline_comparison

    if model_level_metrics is None:
        model_level_metrics = {}

    return ExtendedBusinessOptimizationSummary(
        objective="minimize_expected_business_cost_with_retention",
        selected_threshold=result.selected_threshold,
        cost_assumptions={
            "cost_tn": costs.cost_tn,
            "cost_fp": costs.cost_fp,
            "cost_fn": costs.cost_fn,
            "cost_intervention": costs.cost_intervention,
            "retention_success_rate": costs.retention_success_rate,
            "clv": costs.clv,
            "label": costs.label,
        },
        optimization_summary={
            "tie_breaking_rule": result.tie_breaking_rule,
            "threshold_grid_step": THRESHOLD_STEP,
            "n_thresholds_evaluated": len(result.all_threshold_results),
        },
        business_metrics={
            "total_cost": metrics["total_cost"],
            "average_cost_per_customer": metrics["average_cost"],
            "cost_per_predicted_churn": metrics["cost_per_predicted_churn"],
            "net_business_value": metrics["net_business_value"],
            "expected_churn_loss": metrics["expected_churn_loss"],
            "expected_intervention_cost": metrics["expected_intervention_cost"],
            "expected_retained_value": metrics["expected_retained_value"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "specificity": metrics["specificity"],
            "npv": metrics["npv"],
        },
        baseline_comparison={
            "contact_no_one_total_cost": baseline["contact_no_one_total_cost"],
            "contact_no_one_avg_cost": baseline["contact_no_one_avg_cost"],
            "contact_everyone_total_cost": baseline["contact_everyone_total_cost"],
            "contact_everyone_avg_cost": baseline["contact_everyone_avg_cost"],
            "value_vs_no_contact": baseline["contact_no_one_total_cost"]
            - metrics["total_cost"],
            "value_vs_contact_all": baseline["contact_everyone_total_cost"]
            - metrics["total_cost"],
        },
        model_level_metrics={
            "roc_auc": model_level_metrics.get("roc_auc", 0.0),
            "pr_auc": model_level_metrics.get("pr_auc", 0.0),
        },
        confusion_matrix={
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        },
        tie_breaking_rule=result.tie_breaking_rule,
        optimization_data={
            "data_source": result.optimization_data_source,
            "test_set_used_for_optimization": False,
            "cv_config": {
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42,
            },
        },
    )


def save_extended_analysis(
    result: ExtendedOptimizationResult,
    sensitivity_results: list[dict[str, Any]],
    model_level_metrics: dict[str, float] | None,
    output_dir: Path,
) -> dict[str, Path]:
    """Save all extended business cost analysis artifacts.

    Artifacts created:
        - extended_cost_analysis.json: Full optimization result for primary scenario
        - extended_cost_thresholds.csv: Threshold grid with metrics for primary scenario
        - extended_cost_sensitivity.csv: Sensitivity analysis across scenarios

    Args:
        result: Optimization result for primary scenario.
        sensitivity_results: List of sensitivity analysis results.
        model_level_metrics: Threshold-independent metrics.
        output_dir: Directory to save artifacts (typically models/).

    Returns:
        Dict mapping artifact name to Path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    # Primary scenario JSON
    summary = build_extended_optimization_summary(result, model_level_metrics)
    json_path = output_dir / "extended_cost_analysis.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    artifacts["json"] = json_path

    # Primary scenario CSV
    csv_path = output_dir / "extended_cost_thresholds.csv"
    with open(csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
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
                "net_business_value",
                "expected_churn_loss",
                "expected_intervention_cost",
                "expected_retained_value",
            ]
        )
        for r in result.all_threshold_results:
            csv_writer.writerow(
                [
                    f"{r['threshold']:.2f}",
                    r["tn"],
                    r["fp"],
                    r["fn"],
                    r["tp"],
                    f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}",
                    f"{r['f1']:.6f}",
                    f"{r['accuracy']:.6f}",
                    f"{r['specificity']:.6f}",
                    f"{r['npv']:.6f}",
                    f"{r['total_cost']:.2f}",
                    f"{r['average_cost']:.6f}",
                    f"{r['cost_per_predicted_churn']:.6f}",
                    f"{r['net_business_value']:.2f}",
                    f"{r['expected_churn_loss']:.2f}",
                    f"{r['expected_intervention_cost']:.2f}",
                    f"{r['expected_retained_value']:.2f}",
                ]
            )
    artifacts["csv"] = csv_path

    # Sensitivity analysis CSV
    sensitivity_path = output_dir / "extended_cost_sensitivity.csv"
    if sensitivity_results:
        fieldnames: list[str] = list(sensitivity_results[0].keys())
        with open(sensitivity_path, "w", newline="") as f:
            sensitivity_writer: csv.DictWriter[str] = csv.DictWriter(
                f, fieldnames=fieldnames
            )
            sensitivity_writer.writeheader()
            sensitivity_writer.writerows(sensitivity_results)
    artifacts["sensitivity"] = sensitivity_path

    return artifacts


def plot_extended_cost_curve(
    result: ExtendedOptimizationResult,
    output_path: Path,
    show_secondary: bool = True,
) -> Path:
    """Generate extended business cost vs threshold curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in result.all_threshold_results]
    total_costs = [r["total_cost"] for r in result.all_threshold_results]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        thresholds,
        total_costs,
        "b-",
        linewidth=2,
        label="Total Business Cost (Extended)",
    )
    ax1.axvline(
        result.selected_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Selected Threshold = {result.selected_threshold:.2f}",
    )
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Total Expected Business Cost ($)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    if show_secondary:
        ax2 = ax1.twinx()
        recall_vals = [r["recall"] for r in result.all_threshold_results]
        precision_vals = [r["precision"] for r in result.all_threshold_results]
        f1_vals = [r["f1"] for r in result.all_threshold_results]
        net_value_vals = [r["net_business_value"] for r in result.all_threshold_results]

        ax2.plot(thresholds, recall_vals, "g--", alpha=0.6, label="Recall (OOF)")
        ax2.plot(thresholds, precision_vals, "m--", alpha=0.6, label="Precision (OOF)")
        ax2.plot(thresholds, f1_vals, "c--", alpha=0.6, label="F1 (OOF)")
        ax2.plot(thresholds, net_value_vals, "r--", alpha=0.6, label="Net Value (OOF)")
        ax2.set_ylabel("Secondary Metrics (OOF)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax1.legend(loc="upper left")

    costs = result.cost_matrix
    ax1.set_title(
        "Extended Business Cost vs Threshold (OOF Optimization)\n"
        f"r={costs.retention_success_rate}, CLV={costs.clv}, "
        f"Intervention={costs.cost_intervention}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def evaluate_on_test_with_extended_threshold(
    selected_threshold: float,
    costs: ExtendedCostMatrix,
) -> ExtendedBusinessCostResult:
    """Evaluate the extended business-optimal threshold on the frozen test set.

    This function is ONLY called AFTER threshold selection on OOF data.
    It scores the test set with the frozen, pre-selected threshold.

    The test set is NOT used for threshold selection.

    Args:
        selected_threshold: Business-optimal threshold from OOF optimization.
        costs: ExtendedCostMatrix (same as used for optimization).

    Returns:
        ExtendedBusinessCostResult with test-set metrics.
    """
    from .features import TARGET_COL, load_data
    from .final_evaluation import build_final_pipeline

    train_df = load_data("train")
    test_df = load_data("test")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = np.asarray(
        train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    )
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = np.asarray(
        test_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    )

    pipeline = build_final_pipeline()
    print(f"Fitting final pipeline on {len(X_train)} training samples...")
    pipeline.fit(X_train, y_train)

    print(f"Scoring frozen test set ({len(X_test)} samples)...")
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    return compute_extended_business_cost_metrics(
        y_test, y_proba_test, selected_threshold, costs
    )


def run_phase11_extended(
    costs: ExtendedCostMatrix | None = None,
    output_dir: Path | None = None,
    sensitivity_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run complete Phase 11 Extended business cost optimization pipeline.

    Steps:
        1. Optimize threshold on OOF predictions (training data only).
        2. Evaluate on frozen test set (threshold already selected).
        3. Run sensitivity analysis on OOF data.
        4. Save all artifacts.

    Args:
        costs: Optional ExtendedCostMatrix. Defaults to r=0.5, clv=100, int=10.
        output_dir: Output directory for artifacts. Defaults to models/.
        sensitivity_config: Optional dict with keys retention_rates, clv_values,
            intervention_costs for sensitivity analysis.

    Returns:
        Dictionary with all results and artifact paths.
    """
    if costs is None:
        costs = ExtendedCostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_intervention=10.0,
            retention_success_rate=0.5,
            clv=100.0,
            label="extended default: r=0.5, clv=100, int=10",
        )

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(__file__).resolve().parents[2] / "notebooks" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if sensitivity_config is None:
        sensitivity_config = {
            "retention_rates": [0.2, 0.4, 0.6, 0.8],
            "clv_values": [100.0],
            "intervention_costs": [10.0],
        }

    print("=" * 80)
    print("PHASE 11 EXTENDED - BUSINESS COST OPTIMIZATION WITH RETENTION RATE")
    print("=" * 80)
    print("\nIMPORTANT: Retention success rate, CLV, and intervention cost are")
    print(
        "EXTERNAL BUSINESS ASSUMPTIONS. The dataset contains no intervention history."
    )
    print("=" * 80)

    # Step 1: OOF optimization
    print("\n[1] Optimizing threshold on OOF predictions (training data only)...")
    result = run_extended_optimization(costs)

    # Get model-level metrics on test set for the summary
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    from .features import TARGET_COL, load_data
    from .final_evaluation import build_final_pipeline

    train_df = load_data("train")
    test_df = load_data("test")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = np.asarray(
        train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    )
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = np.asarray(
        test_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    )

    pipeline = build_final_pipeline()
    pipeline.fit(X_train, y_train)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    model_level_metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba_test)),
        "pr_auc": float(average_precision_score(y_test, y_proba_test)),
    }

    # Step 2: Evaluate on frozen test set
    print("\n[2] Evaluating extended business-optimal threshold on frozen test set...")
    test_metrics = evaluate_on_test_with_extended_threshold(
        result.selected_threshold, costs
    )

    # Step 3: Sensitivity analysis on OOF
    print("\n[3] Running sensitivity analysis on OOF data...")
    from .threshold import get_oof_predictions

    y_true_oof, y_proba_oof = get_oof_predictions()
    sensitivity = run_sensitivity_analysis(
        y_true_oof,
        y_proba_oof,
        retention_rates=sensitivity_config.get("retention_rates"),
        clv_values=sensitivity_config.get("clv_values"),
        intervention_costs=sensitivity_config.get("intervention_costs"),
    )

    # Step 4: Save artifacts
    print("\n[4] Saving artifacts...")
    artifact_paths = save_extended_analysis(
        result, sensitivity, model_level_metrics, output_dir
    )

    # Save cost curve
    curve_path = fig_dir / "extended_cost_curve.png"
    plot_extended_cost_curve(result, curve_path)
    artifact_paths["figure"] = curve_path

    # Build test evaluation summary
    summary = build_extended_optimization_summary(result, model_level_metrics)

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 11 EXTENDED RESULTS")
    print("=" * 80)
    print(f"\nCost Assumptions ({costs.label}):")
    print(
        f"  Cost(TN)={costs.cost_tn}, Cost(FP)={costs.cost_fp}, "
        f"Cost(FN)={costs.cost_fn}, Cost(Intervention)={costs.cost_intervention}"
    )
    print(f"  Retention Success Rate (r)={costs.retention_success_rate}")
    print(f"  CLV={costs.clv}")

    print(f"\nOOF-Optimal Extended Business Threshold: {result.selected_threshold:.2f}")
    print(f"  Total Cost (OOF):     ${result.selected_metrics['total_cost']:.2f}")
    print(f"  Avg Cost/Customer:    ${result.selected_metrics['average_cost']:.2f}")
    print(
        f"  Net Business Value:   ${result.selected_metrics['net_business_value']:.2f}"
    )
    print(f"  Precision (OOF):      {result.selected_metrics['precision']:.4f}")
    print(f"  Recall (OOF):         {result.selected_metrics['recall']:.4f}")
    print(f"  F1 (OOF):             {result.selected_metrics['f1']:.4f}")

    print(f"\nFrozen Test Evaluation at Threshold={result.selected_threshold:.2f}:")
    print(f"  Total Cost (Test):    ${test_metrics['total_cost']:.2f}")
    print(f"  Avg Cost/Customer:    ${test_metrics['average_cost']:.2f}")
    print(f"  Net Business Value:   ${test_metrics['net_business_value']:.2f}")
    print(f"  Precision (Test):     {test_metrics['precision']:.4f}")
    print(f"  Recall (Test):        {test_metrics['recall']:.4f}")
    print(f"  F1 (Test):            {test_metrics['f1']:.4f}")
    print(f"  Accuracy (Test):      {test_metrics['accuracy']:.4f}")
    print(
        f"  Confusion Matrix:     TN={test_metrics['tn']}, "
        f"FP={test_metrics['fp']}, FN={test_metrics['fn']}, "
        f"TP={test_metrics['tp']}"
    )

    baseline = result.baseline_comparison
    print("\nBaseline Comparisons (OOF population):")
    print(f"  Contact No One:       ${baseline['contact_no_one_total_cost']:.2f}")
    print(f"  Contact Everyone:     ${baseline['contact_everyone_total_cost']:.2f}")
    print(
        f"  Value vs No Contact:  ${baseline['contact_no_one_total_cost'] - result.selected_metrics['total_cost']:.2f}"
    )
    print(
        f"  Value vs Contact All: ${baseline['contact_everyone_total_cost'] - result.selected_metrics['total_cost']:.2f}"
    )

    print("\nSensitivity Analysis (OOF):")
    print(
        f"  {'Scenario':<50} {'r':>4} {'Threshold':>10} {'Cost':>10} {'Value vs None':>12}"
    )
    print(f"  {'-' * 50} {'-' * 4} {'-' * 10} {'-' * 10} {'-' * 12}")
    for s in sensitivity:
        print(
            f"  {s['scenario_label']:<50} {s['retention_success_rate']:>4.1f} "
            f"{s['optimal_threshold']:>10.2f} {s['optimal_total_cost']:>10.2f} "
            f"{s['value_vs_no_contact']:>12.2f}"
        )

    print("\nArtifacts saved:")
    for name, path in artifact_paths.items():
        print(f"  {name}: {path}")

    print(f"\nTie-breaking rule: {result.tie_breaking_rule}")

    print("\nNOTE: Phase 11 production threshold was 0.28 (r=1.0 implicit).")
    print(
        f"      Extended optimal threshold at r={costs.retention_success_rate} is {result.selected_threshold:.2f}."
    )
    print(
        "      Retention success rate, CLV, and intervention cost are EXTERNAL ASSUMPTIONS."
    )

    return {
        "optimization_result": result,
        "model_level_metrics": model_level_metrics,
        "test_metrics": test_metrics,
        "sensitivity_analysis": sensitivity,
        "artifacts": artifact_paths,
        "summary": summary,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_phase11_extended()

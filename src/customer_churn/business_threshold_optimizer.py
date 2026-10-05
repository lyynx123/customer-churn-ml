"""Business Threshold Optimizer - Phase 11.

Searches for the probability threshold that minimizes expected business cost
on OOF (out-of-fold) predictions.

LEAKAGE RULE:
    This optimizer ONLY accepts OOF / validation probabilities.
    Frozen test set probabilities MUST NOT be passed to the optimizer.
    The frozen test set is only used AFTER threshold is selected for final
    evaluation.

The OOF predictions come from `threshold.get_oof_predictions()` which
uses cross_val_predict on training data only.
"""

import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from .business_cost import (
    BusinessCostResult,
    CostMatrix,
    compute_business_cost_metrics,
    validate_costs,
)

THRESHOLD_STEP: float = 0.01


@dataclass
class OptimizationResult:
    """Result of business threshold optimization."""

    selected_threshold: float
    selected_metrics: BusinessCostResult
    cost_matrix: CostMatrix
    all_threshold_results: list[BusinessCostResult] = field(default_factory=list)
    optimization_data_source: str = "OOF (cross-validation on training data)"
    test_set_used_for_optimization: bool = False
    tie_breaking_rule: str = field(
        default="1. minimum total_cost, 2. higher recall, 3. lower threshold"
    )


class BusinessOptimizationSummary(TypedDict):
    """Machine-readable summary for JSON artifact."""

    objective: str
    selected_threshold: float
    cost_assumptions: dict[str, Any]
    optimization_summary: dict[str, Any]
    business_metrics: dict[str, Any]
    model_level_metrics: dict[str, Any]
    confusion_matrix: dict[str, int]
    tie_breaking_rule: str
    optimization_data: dict[str, Any]


def build_business_threshold_grid(
    start: float = 0.01,
    end: float = 0.99,
    step: float = THRESHOLD_STEP,
) -> list[float]:
    """Build deterministic threshold grid for optimization.

    Args:
        start: Minimum threshold (default 0.01).
        end: Maximum threshold (default 0.99).
        step: Grid step (default 0.01).

    Returns:
        Sorted list of thresholds.

    Raises:
        ValueError: If start/end/step are invalid.
    """
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
    results: list[BusinessCostResult], costs: CostMatrix, tolerance: float = 1e-6
) -> BusinessCostResult:
    """Select optimal result using deterministic tie-breaking.

    Updated tie-breaking rule (documented and deterministic):
        1. Minimum total_cost.
        2. If tied and FP cost > FN cost: choose higher threshold (reduces FP).
        3. If tied and FP cost <= FN cost: choose higher recall.
        4. If still tied: choose lower threshold.

    Args:
        results: List of threshold evaluation results.
        costs: Business cost matrix used for optimization.
        tolerance: Numerical tolerance for cost equality.

    Returns:
        The best BusinessCostResult.
    """
    min_cost = min(r["total_cost"] for r in results)
    tied = [r for r in results if abs(r["total_cost"] - min_cost) < tolerance]

    if len(tied) == 1:
        return tied[0]

    # Cost asymmetry check
    if costs.cost_fp > costs.cost_fn:
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


def optimize_threshold_business(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.float64],
    costs: CostMatrix,
    threshold_grid: list[float] | None = None,
) -> OptimizationResult:
    """Find threshold minimizing expected business cost on OOF predictions.

    Args:
        y_true: OOF true labels (must be from training-data CV, NOT test).
        y_proba: OOF predicted probabilities (must be from training-data CV,
            NOT test).
        costs: Business cost matrix.
        threshold_grid: Optional custom threshold grid. Defaults to
            0.01 to 0.99 with 0.01 step.

    Returns:
        OptimizationResult with selected threshold and all evaluation results.

    Raises:
        ValueError: If inputs are invalid.
    """
    validate_costs(costs)

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

    all_results: list[BusinessCostResult] = [
        compute_business_cost_metrics(y_true, y_proba, t, costs) for t in threshold_grid
    ]

    selected = _tie_break(all_results, costs)

    return OptimizationResult(
        selected_threshold=selected["threshold"],
        selected_metrics=selected,
        all_threshold_results=all_results,
        cost_matrix=costs,
    )


def run_business_optimization(
    costs: CostMatrix | None = None,
) -> OptimizationResult:
    """Run full business threshold optimization using OOF predictions.

    Uses `threshold.get_oof_predictions()` which generates OOF predictions
    via cross_val_predict on training data only. NO test set involvement.

    Args:
        costs: Optional cost matrix. Defaults to illustrative default.

    Returns:
        OptimizationResult with selected business-optimal threshold.
    """
    from .threshold import get_oof_predictions

    if costs is None:
        costs = CostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_tp=10.0,
            label="illustrative default: FN=10x FP",
        )

    y_true_oof, y_proba_oof = get_oof_predictions()

    result = optimize_threshold_business(y_true_oof, y_proba_oof, costs)
    result.optimization_data_source = (
        "OOF (5-fold StratifiedKFold cross_val_predict on training data)"
    )
    return result


def build_sensitivity_scenarios() -> list[CostMatrix]:
    """Build illustrative cost sensitivity scenarios.

    These scenarios demonstrate how the business-optimal threshold changes
    with different FN/FP cost ratios.

    All values are illustrative assumptions, NOT real business data.

    Returns:
        List of CostMatrix scenarios for sensitivity analysis.
    """
    scenarios: list[CostMatrix] = [
        CostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=10.0,
            cost_tp=10.0,
            label="Scenario A: FN and FP balanced (1:1)",
        ),
        CostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=50.0,
            cost_tp=10.0,
            label="Scenario B: FN moderately more expensive (5x FP)",
        ),
        CostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_tp=10.0,
            label="Scenario C: FN more expensive (10x FP)",
        ),
        CostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=200.0,
            cost_tp=10.0,
            label="Scenario D: FN very expensive (20x FP)",
        ),
        CostMatrix(
            cost_tn=0.0,
            cost_fp=50.0,
            cost_fn=100.0,
            cost_tp=50.0,
            label="Scenario E: FP more expensive than FN (0.5x)",
        ),
    ]
    return scenarios


def run_sensitivity_analysis(
    y_true_oof: NDArray[np.int_],
    y_proba_oof: NDArray[np.float64],
    scenarios: list[CostMatrix] | None = None,
) -> list[dict[str, Any]]:
    """Run cost sensitivity analysis across illustrative scenarios.

    For each cost scenario, finds the business-optimal threshold on OOF data.

    Args:
        y_true_oof: OOF true labels (training data only).
        y_proba_oof: OOF probabilities (training data only).
        scenarios: Optional list of cost scenarios. Defaults to built-in set.

    Returns:
        List of scenario results with cost assumptions and optimal thresholds.
    """
    if scenarios is None:
        scenarios = build_sensitivity_scenarios()

    results: list[dict[str, Any]] = []

    for scenario in scenarios:
        opt = optimize_threshold_business(y_true_oof, y_proba_oof, scenario)
        fn_cost = scenario.cost_fn
        fp_cost = scenario.cost_fp
        results.append(
            {
                "scenario_label": scenario.label,
                "cost_tn": scenario.cost_tn,
                "cost_fp": scenario.cost_fp,
                "cost_fn": scenario.cost_fn,
                "cost_tp": scenario.cost_tp,
                "fn_to_fp_ratio": fn_cost / fp_cost if fp_cost > 0 else float("inf"),
                "optimal_threshold": opt.selected_threshold,
                "optimal_total_cost": opt.selected_metrics["total_cost"],
                "optimal_average_cost": opt.selected_metrics["average_cost"],
                "o1_optimal_recall": opt.selected_metrics["recall"],
                "optimal_precision": opt.selected_metrics["precision"],
                "optimal_f1": opt.selected_metrics["f1"],
            }
        )

    return results


def build_optimization_summary(
    result: OptimizationResult,
    model_level_metrics: dict[str, float] | None = None,
) -> BusinessOptimizationSummary:
    """Build machine-readable summary for JSON artifact.

    Args:
        result: Optimization result from run_business_optimization.
        model_level_metrics: Optional threshold-independent metrics
            (e.g., ROC-AUC, PR-AUC).

    Returns:
        TypedDict for JSON serialization.
    """
    metrics = result.selected_metrics
    costs = result.cost_matrix

    if model_level_metrics is None:
        model_level_metrics = {}

    return BusinessOptimizationSummary(
        objective="minimize_expected_business_cost",
        selected_threshold=result.selected_threshold,
        cost_assumptions={
            "cost_tn": costs.cost_tn,
            "cost_fp": costs.cost_fp,
            "cost_fn": costs.cost_fn,
            "cost_tp": costs.cost_tp,
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
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "specificity": metrics["specificity"],
            "npv": metrics["npv"],
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


def save_business_analysis(
    result: OptimizationResult,
    model_level_metrics: dict[str, float] | None,
    output_dir: Path,
) -> dict[str, Path]:
    """Save all business cost analysis artifacts.

    Artifacts created:
        - business_cost_analysis.json: Full optimization result
        - business_cost_thresholds.csv: Threshold grid with metrics

    Args:
        result: Optimization result.
        model_level_metrics: Threshold-independent metrics.
        output_dir: Directory to save artifacts (typically models/).

    Returns:
        Dict mapping artifact name to Path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    summary = build_optimization_summary(result, model_level_metrics)

    json_path = output_dir / "business_cost_analysis.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    artifacts["json"] = json_path

    csv_path = output_dir / "business_cost_thresholds.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
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
            ]
        )
        for r in result.all_threshold_results:
            writer.writerow(
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
                ]
            )
    artifacts["csv"] = csv_path

    return artifacts


def plot_cost_curve(
    result: OptimizationResult,
    output_path: Path,
    show_secondary: bool = True,
) -> Path:
    """Generate business cost vs threshold curve.

    Args:
        result: Optimization result with full threshold grid.
        output_path: Where to save the figure.
        show_secondary: Whether to overlay secondary metrics (recall, precision,
            F1) on secondary y-axis.

    Returns:
        Path to the saved figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in result.all_threshold_results]
    costs = [r["total_cost"] for r in result.all_threshold_results]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(thresholds, costs, "b-", linewidth=2, label="Total Business Cost")
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

        ax2.plot(thresholds, recall_vals, "g--", alpha=0.6, label="Recall (OOF)")
        ax2.plot(
            thresholds, precision_vals, "m--", alpha=0.6, label="Precision (OOF)"
        )
        ax2.plot(thresholds, f1_vals, "c--", alpha=0.6, label="F1 (OOF)")
        ax2.set_ylabel("Secondary Metrics (OOF)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax1.legend(loc="upper left")

    ax1.set_title(
        "Business Cost vs Threshold (OOF Optimization)\n"
        f"Cost Matrix: TN={result.cost_matrix.cost_tn}, "
        f"FP={result.cost_matrix.cost_fp}, "
        f"FN={result.cost_matrix.cost_fn}, "
        f"TP={result.cost_matrix.cost_tp}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def evaluate_on_test_with_business_threshold(
    selected_threshold: float,
    costs: CostMatrix,
) -> BusinessCostResult:
    """Evaluate the business-optimal threshold on the frozen test set.

    This function is ONLY called AFTER threshold selection on OOF data.
    It scores the test set with the frozen, pre-selected threshold.

    The test set is NOT used for threshold selection.

    Args:
        selected_threshold: Business-optimal threshold from OOF optimization.
        costs: Business cost matrix (same as used for optimization).

    Returns:
        BusinessCostResult with test-set metrics.
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

    return compute_business_cost_metrics(y_test, y_proba_test, selected_threshold, costs)


def run_phase11(
    costs: CostMatrix | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run complete Phase 11 business cost optimization pipeline.

    Steps:
        1. Optimize threshold on OOF predictions (training data only).
        2. Evaluate on frozen test set (threshold already selected).
        3. Run sensitivity analysis on OOF data.
        4. Save all artifacts.

    Args:
        costs: Optional cost matrix. Defaults to illustrative scenario.
        output_dir: Output directory for artifacts. Defaults to models/.

    Returns:
        Dictionary with all results and artifact paths.
    """
    if costs is None:
        costs = CostMatrix(
            cost_tn=0.0,
            cost_fp=10.0,
            cost_fn=100.0,
            cost_tp=10.0,
            label="illustrative default: FN=10x FP",
        )

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(__file__).resolve().parents[2] / "notebooks" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 11 - BUSINESS COST OPTIMIZATION")
    print("=" * 80)

    # Step 1: OOF optimization
    print("\n[1] Optimizing threshold on OOF predictions (training data only)...")
    result = run_business_optimization(costs)

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
    print("\n[2] Evaluating business-optimal threshold on frozen test set...")
    test_metrics = evaluate_on_test_with_business_threshold(
        result.selected_threshold, costs
    )

    # Step 3: Sensitivity analysis on OOF
    print("\n[3] Running cost sensitivity analysis on OOF data...")
    from .threshold import get_oof_predictions

    y_true_oof, y_proba_oof = get_oof_predictions()
    sensitivity = run_sensitivity_analysis(y_true_oof, y_proba_oof)

    # Step 4: Save artifacts
    print("\n[4] Saving artifacts...")
    artifact_paths = save_business_analysis(result, model_level_metrics, output_dir)

    # Save cost curve
    curve_path = fig_dir / "business_cost_curve.png"
    plot_cost_curve(result, curve_path)
    artifact_paths["figure"] = curve_path

    # Save sensitivity analysis
    sensitivity_path = output_dir / "cost_sensitivity_analysis.csv"
    with open(sensitivity_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity)
    artifact_paths["sensitivity"] = sensitivity_path

    # Build test evaluation summary
    summary = build_optimization_summary(result, model_level_metrics)

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 11 RESULTS")
    print("=" * 80)
    print(f"\nCost Matrix ({costs.label}):")
    print(f"  Cost(TN)={costs.cost_tn}, Cost(FP)={costs.cost_fp}, "
          f"Cost(FN)={costs.cost_fn}, Cost(TP)={costs.cost_tp}")

    print(f"\nOOF-Optimal Business Threshold: {result.selected_threshold:.2f}")
    print(f"  Total Cost (OOF):     ${result.selected_metrics['total_cost']:.2f}")
    print(f"  Avg Cost/Customer:    ${result.selected_metrics['average_cost']:.2f}")
    print(f"  Precision (OOF):      {result.selected_metrics['precision']:.4f}")
    print(f"  Recall (OOF):         {result.selected_metrics['recall']:.4f}")
    print(f"  F1 (OOF):             {result.selected_metrics['f1']:.4f}")
    print(f"  Specificity (OOF):    {result.selected_metrics['specificity']:.4f}")

    print(f"\nFrozen Test Evaluation at Threshold={result.selected_threshold:.2f}:")
    print(f"  Total Cost (Test):    ${test_metrics['total_cost']:.2f}")
    print(f"  Avg Cost/Customer:    ${test_metrics['average_cost']:.2f}")
    print(f"  Precision (Test):     {test_metrics['precision']:.4f}")
    print(f"  Recall (Test):        {test_metrics['recall']:.4f}")
    print(f"  F1 (Test):            {test_metrics['f1']:.4f}")
    print(f"  Accuracy (Test):      {test_metrics['accuracy']:.4f}")
    print(f"  Confusion Matrix:     TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']}, "
          f"TP={test_metrics['tp']}")

    print("\nSensitivity Analysis (OOF):")
    print(f"  {'Scenario':<40} {'FN/FP':>8} {'Threshold':>12} {'Cost':>12}")
    print(f"  {'-'*40} {'-'*8} {'-'*12} {'-'*12}")
    for s in sensitivity:
        print(f"  {s['scenario_label']:<40} {s['fn_to_fp_ratio']:>8.1f} "
              f"{s['optimal_threshold']:>12.2f} {s['optimal_total_cost']:>12.2f}")

    print("\nArtifacts saved:")
    for name, path in artifact_paths.items():
        print(f"  {name}: {path}")

    print(f"\nTie-breaking rule: {result.tie_breaking_rule}")

    print("\nNOTE: F1-optimal threshold from Phase 8B was 0.55.")
    print(f"      Business-optimal threshold from Phase 11 is {result.selected_threshold:.2f}.")
    print("      The business threshold is optimal for the specified cost matrix:")
    print(f"      {costs.label}.")
    print("      These differ because business cost optimization minimizes expected")
    print("      financial loss, not F1 score.")

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
    run_phase11()

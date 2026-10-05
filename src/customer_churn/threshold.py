"""Threshold optimization module for Customer Churn Prediction.

This module performs leakage-safe decision threshold analysis using
out-of-fold predictions from cross-validation on training data only.
"""

import json
import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import RANDOM_SEED
from .features import TARGET_COL, build_preprocessor, load_data
from .models import build_model_pipeline

# Tuned RandomForest configuration from Phase 8A
TUNED_RF_CONFIG: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}


def build_tuned_model() -> RandomForestClassifier:
    """Construct the tuned RandomForestClassifier from Phase 8A parameters."""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


# Threshold grid for analysis
THRESHOLD_GRID = np.arange(0.10, 0.95, 0.05).round(2).tolist()


def get_oof_predictions() -> tuple[np.ndarray, np.ndarray]:  # type: ignore[return-value]
    """Generate out-of-fold probability predictions using CV.

    Returns:
        Tuple of (y_true, y_proba_oof) where y_proba_oof are OOF probabilities
        for the positive class (churn=1).
    """
    train_df = load_data("train")
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    model = build_tuned_model()
    preprocessor = build_preprocessor()
    pipeline = build_model_pipeline(model, preprocessor)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Get OOF probabilities for positive class
    y_proba_oof = cross_val_predict(
        pipeline,
        X,
        y,  # type: ignore[arg-type]
        cv=cv,
        method="predict_proba",
        n_jobs=2,
    )[:, 1]

    return y, y_proba_oof  # type: ignore[return-value]


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute all metrics for a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Derived metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    pos_rate = (tp + fp) / len(y_true)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "positive_rate": float(pos_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_threshold_analysis() -> list[dict[str, float]]:
    """Run complete threshold analysis using OOF predictions."""
    y_true, y_proba_oof = get_oof_predictions()

    results = []
    for thresh in THRESHOLD_GRID:
        metrics = compute_threshold_metrics(y_true, y_proba_oof, thresh)
        results.append(metrics)

    return results


def find_optimal_thresholds(results: list[dict[str, float]]) -> dict[str, float | None]:
    """Identify candidate optimal thresholds based on different criteria."""

    # Default threshold 0.50
    default = next((r for r in results if abs(r["threshold"] - 0.50) < 1e-6), None)

    # Max F1
    f1_opt = max(results, key=lambda r: r["f1"])

    # Max Recall with Precision >= 0.50 (if exists)
    recall_candidates = [r for r in results if r["precision"] >= 0.50]
    recall_opt = (
        max(recall_candidates, key=lambda r: r["recall"]) if recall_candidates else None
    )

    # Max Precision with Recall >= 0.60 (if exists)
    prec_candidates = [r for r in results if r["recall"] >= 0.60]
    prec_opt = (
        max(prec_candidates, key=lambda r: r["precision"]) if prec_candidates else None
    )

    return {
        "default": default["threshold"] if default else None,
        "f1_optimal": f1_opt["threshold"],
        "recall_oriented": recall_opt["threshold"] if recall_opt else None,
        "precision_oriented": prec_opt["threshold"] if prec_opt else None,
    }  # type: ignore[return-value]


def format_threshold_table(results: list[dict[str, float]]) -> str:
    """Format threshold analysis as a markdown table."""
    lines = []
    lines.append(
        "| Threshold | Precision | Recall | F1 | Accuracy | FPR | FNR | Positive Rate |"
    )
    lines.append(
        "| --------: | --------: | -----: | -: | -------: | --: | --: | ------------: |"
    )

    for r in results:
        lines.append(
            f"| {r['threshold']:.2f} | "
            f"{r['precision']:.4f} | "
            f"{r['recall']:.4f} | "
            f"{r['f1']:.4f} | "
            f"{r['accuracy']:.4f} | "
            f"{r['fpr']:.4f} | "
            f"{r['fnr']:.4f} | "
            f"{r['positive_rate']:.4f} |"
        )
    return "\n".join(lines)


def save_analysis_results(
    results: list[dict[str, float]],
    optimal_thresholds: dict[str, float | None],
    output_path: str,
):
    """Save threshold analysis results to JSON."""
    output = {
        "model_config": TUNED_RF_CONFIG,
        "cv_config": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": RANDOM_SEED,
        },
        "threshold_grid": THRESHOLD_GRID,
        "threshold_results": results,
        "optimal_thresholds": optimal_thresholds,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def plot_threshold_curves(results: list[dict[str, float]], output_path: str):
    """Create threshold vs metrics visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [r["threshold"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precision, label="Precision", marker="o")
    ax.plot(thresholds, recall, label="Recall", marker="s")
    ax.plot(thresholds, f1, label="F1", marker="^")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Threshold vs Metrics (OOF CV on Training Data)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(THRESHOLD_GRID)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("Running threshold analysis (OOF CV on training data)...")
    print("=" * 80)

    # Run analysis
    results = run_threshold_analysis()
    optimal = find_optimal_thresholds(results)

    # Print table
    print("\nThreshold Analysis (OOF CV on Training Data):")
    print("=" * 80)
    print(format_threshold_table(results))

    print("\nOptimal Threshold Candidates:")
    print("=" * 80)
    for name, thresh in optimal.items():
        if thresh is not None:
            r = next(r for r in results if abs(r["threshold"] - thresh) < 1e-6)
            print(
                f"  {name}: {thresh:.2f} (F1={r['f1']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f})"
            )
        else:
            print(f"  {name}: Not available")

    # Save artifacts
    output_path = "/home/ahmad_izzuddin_ulinnuha/projects/Customer_Churn/models/threshold_analysis.json"
    save_analysis_results(results, optimal, output_path)

    figure_path = "/home/ahmad_izzuddin_ulinnuha/projects/Customer_Churn/notebooks/figures/threshold_analysis_cv.png"
    plot_threshold_curves(results, figure_path)

    print("\nArtifacts saved:")
    print(f"  JSON: {output_path}")
    print(f"  Figure: {figure_path}")

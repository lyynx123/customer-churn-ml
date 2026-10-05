"""Phase 8C - Final Held-Out Test Evaluation.

This module evaluates the tuned RandomForestClassifier and pre-specified
threshold candidates on the held-out test set.
"""

import json
import warnings
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import RANDOM_SEED
from .features import TARGET_COL, build_preprocessor, load_data
from .models import build_model_pipeline

# Tuned RandomForest configuration from Phase 8A/8B
FINAL_RF_CONFIG = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Threshold candidates from Phase 8B
THRESHOLD_CANDIDATES = [0.45, 0.50, 0.55, 0.60]


class ThresholdMetrics(TypedDict):
    """Threshold-specific evaluation metrics (counts included as ints)."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    fpr: float
    fnr: float
    positive_rate: float
    tn: int
    fp: int
    fn: int
    tp: int


class ModelLevelMetrics(TypedDict):
    """Threshold-independent ranking metrics."""

    roc_auc: float
    pr_auc: float


class FinalEvaluation(TypedDict):
    """Full Phase 8C evaluation result, as persisted to JSON."""

    metadata: dict[str, Any]
    model_level_metrics: ModelLevelMetrics
    threshold_evaluations: dict[str, ThresholdMetrics]


def build_final_pipeline() -> Any:
    """Build preprocessing + model pipeline."""
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    preprocessor = build_preprocessor()
    return build_model_pipeline(model, preprocessor)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> ThresholdMetrics:
    """Compute all required threshold-specific metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    pos_rate = (tp + fp) / len(y_true)

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "positive_rate": float(pos_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_final_test_evaluation() -> dict[str, Any]:
    """Execute the final evaluation protocol on the held-out test set."""
    # 1. Load data
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

    # 2. Fit pipeline on full training data
    pipeline = build_final_pipeline()
    print(f"Fitting final pipeline on {len(X_train)} training samples...")
    pipeline.fit(X_train, y_train)

    # 3. Predict probabilities on test set
    print(f"Generating predictions for {len(X_test)} test samples...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 4. Model-level metrics (threshold-independent)
    results: FinalEvaluation = {
        "metadata": {
            "model_name": "RandomForestClassifier",
            "parameters": FINAL_RF_CONFIG,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "random_state": RANDOM_SEED,
        },
        "model_level_metrics": {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
        },
        "threshold_evaluations": {},
    }

    # 5. Threshold-level evaluations
    for thresh in THRESHOLD_CANDIDATES:
        y_pred = (y_proba >= thresh).astype(int)
        results["threshold_evaluations"][str(thresh)] = compute_metrics(
            y_test, y_pred, y_proba
        )

    return results  # type: ignore[return-value]


def format_final_table(results: dict[str, Any]) -> str:
    """Format evaluation results as a markdown table."""
    lines = []
    lines.append(
        "| Threshold | Precision | Recall | F1 | Accuracy | FPR | FNR | Pos Rate | TN | FP | FN | TP |"
    )
    lines.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )

    for thresh in THRESHOLD_CANDIDATES:
        r = results["threshold_evaluations"][str(thresh)]
        lines.append(
            f"| {thresh:.2f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
            f"{r['accuracy']:.4f} | {r['fpr']:.4f} | {r['fnr']:.4f} | {r['positive_rate']:.4f} | "
            f"{r['tn']} | {r['fp']} | {r['fn']} | {r['tp']} |"
        )
    return "\n".join(lines)


def save_artifacts(results: dict[str, Any], output_dir: Path):
    """Save evaluation results and generate figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "final_test_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    # Generate curves and CMs if matplotlib is available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # only used for heatmap inside this block

        fig_dir = output_dir.parent / "notebooks" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 1. Threshold comparison plot
        thresh_vals = THRESHOLD_CANDIDATES
        metrics = ["precision", "recall", "f1"]
        plt.figure(figsize=(10, 6))
        for m in metrics:
            plt.plot(
                thresh_vals,
                [results["threshold_evaluations"][str(t)][m] for t in thresh_vals],
                label=m.capitalize(),
                marker="o",
            )
        plt.title("Metrics vs Threshold on Held-Out Test Set")
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(fig_dir / "final_threshold_comparison_test.png", dpi=150)
        plt.close()

        # 2. Confusion matrices
        for thresh in THRESHOLD_CANDIDATES:
            r = results["threshold_evaluations"][str(thresh)]
            cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"Confusion Matrix (Threshold={thresh:.2f})")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.xticks([0.5, 1.5], ["No", "Yes"])
            plt.yticks([0.5, 1.5], ["No", "Yes"])
            filename = (
                f"final_confusion_matrix_threshold_{str(thresh).replace('.', '')}.png"
            )
            plt.savefig(fig_dir / filename, dpi=150)
            plt.close()

        print(f"Figures saved to {fig_dir}")

    except ImportError:
        print("Matplotlib/Seaborn not available, skipping figure generation.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    results = run_final_test_evaluation()

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {results['metadata']['model_name']}")
    print(f"ROC-AUC: {results['model_level_metrics']['roc_auc']:.4f}")
    print(f"PR-AUC:  {results['model_level_metrics']['pr_auc']:.4f}")
    print("\n" + format_final_table(results))
    print("=" * 80)

    save_artifacts(results, Path(__file__).resolve().parents[2] / "models")

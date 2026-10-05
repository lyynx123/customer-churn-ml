"""Phase 7 - Final Model Evaluation on held-out test set."""

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED
from .features import TARGET_COL, build_preprocessor, load_data

# Candidate models with exact Phase 6 configurations
CANDIDATE_MODELS = {
    "LogisticRegression": LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    ),
    "RandomForestClassifier": RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
    ),
}


def build_pipeline(model: Any) -> Pipeline:
    """Build preprocessing + model pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate_on_test(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
) -> dict[str, Any]:
    """Fit on train, predict on test, return metrics."""
    # Fit on training data only
    pipeline.fit(X_train, y_train)

    # Predictions on test
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Confusion matrix (labels: 0=No, 1=Yes)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Classification report
    cls_report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "classification_report": cls_report,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def run_final_evaluation() -> dict[str, Any]:
    """Run final evaluation on held-out test set."""
    # Load data
    train_df = load_data("train")
    test_df = load_data("test")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    results = {}

    for name, model in CANDIDATE_MODELS.items():
        print(f"Evaluating {name} on test set...")
        pipeline = build_pipeline(model)
        eval_result = evaluate_on_test(pipeline, X_train, y_train, X_test, y_test)
        results[name] = eval_result

        # Print summary
        m = eval_result["metrics"]
        cm = eval_result["confusion_matrix"]
        print(f"  ROC-AUC: {m['roc_auc']:.4f}")
        print(f"  PR-AUC:  {m['pr_auc']:.4f}")
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall: {m['recall']:.4f}")
        print(f"  F1: {m['f1']:.4f}")
        print(f"  CM: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

    return results


def format_comparison_table(results: dict[str, Any]) -> str:
    """Format final comparison table."""
    lines = []
    lines.append("| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |")
    lines.append("| ----- | ------: | -----: | -------: | --------: | -----: | -: |")

    for name, result in results.items():
        m = result["metrics"]
        lines.append(
            f"| {name} | "
            f"{m['roc_auc']:.4f} | "
            f"{m['pr_auc']:.4f} | "
            f"{m['accuracy']:.4f} | "
            f"{m['precision']:.4f} | "
            f"{m['recall']:.4f} | "
            f"{m['f1']:.4f} |"
        )
    return "\n".join(lines)


def format_cm_text(cm_dict: dict, model_name: str) -> str:
    """Format confusion matrix as text."""
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    return (
        f"\n{model_name} Confusion Matrix (labels: 0=No, 1=Yes):\n"
        f"              Predicted\n"
        f"              No    Yes\n"
        f"Actual No    {tn:4d}  {fp:4d}\n"
        f"Actual Yes   {fn:4d}  {tp:4d}\n"
        f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
        f"False Positive Rate: {fp / (fp + tn):.4f}\n"
        f"False Negative Rate: {fn / (fn + tp):.4f}\n"
    )


def save_figures(
    results: dict[str, Any],
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    output_dir: Path,
):
    """Save ROC and PR curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, result in results.items():
        y_proba = result["y_proba"]
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)
    ax.set_title("ROC Curves - Test Set")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves_test.png", dpi=150)
    plt.close(fig)

    # PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, result in results.items():
        y_proba = result["y_proba"]
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)
    ax.set_title("Precision-Recall Curves - Test Set")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curves_test.png", dpi=150)
    plt.close(fig)

    # Confusion matrices
    for name, result in results.items():
        cm = result["confusion_matrix"]["matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticklabels(["No", "Yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"cm_{name.lower()}_test.png", dpi=150)
        plt.close(fig)


def save_results_json(results: dict[str, Any], output_path: Path):
    """Save metrics to JSON for reproducibility."""
    output = {}
    for name, result in results.items():
        output[name] = {
            "metrics": result["metrics"],
            "confusion_matrix": result["confusion_matrix"],
            "classification_report": result["classification_report"],
        }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Run evaluation
    results = run_final_evaluation()

    # Print comparison table
    print("\n" + "=" * 80)
    print("FINAL MODEL EVALUATION - HELD-OUT TEST SET")
    print("=" * 80)
    print(format_comparison_table(results))
    print()

    # Print confusion matrices and error analysis
    for name, result in results.items():
        print(format_cm_text(result["confusion_matrix"], name))

    # Save artifacts
    output_dir = Path(__file__).resolve().parents[2] / "notebooks" / "figures"
    train_df = load_data("train")
    test_df = load_data("test")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    save_figures(results, X_train, y_train, X_test, y_test, output_dir)
    save_results_json(
        results,
        Path(__file__).resolve().parents[2] / "models" / "test_evaluation_results.json",
    )

    print(f"\nArtifacts saved to {output_dir}")
    print("JSON results saved to models/test_evaluation_results.json")

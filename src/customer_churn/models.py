"""Baseline modeling module for Customer Churn Prediction.

This module provides baseline classifiers and cross-validation evaluation
using the preprocessing pipeline from features.py.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_SEED
from .features import TARGET_COL, build_preprocessor, load_data

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def get_baseline_models() -> dict[str, Any]:
    """Return dictionary of baseline classifiers with fixed random_state."""
    return {
        "DummyClassifier (prior)": DummyClassifier(
            strategy="prior",
            random_state=RANDOM_SEED,
        ),
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        "RandomForestClassifier": RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def build_model_pipeline(model: Any, preprocessor) -> Pipeline:
    """Build a sklearn Pipeline combining preprocessing and model."""
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model_cv(
    model: Any,
    X: pd.DataFrame,
    y,
    cv_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Evaluate a model using stratified cross-validation.

    Args:
        model: sklearn estimator (will be wrapped in pipeline with preprocessor)
        X: Training features DataFrame
        y: Training target array (0/1)
        cv_splits: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with mean and std of each metric across CV folds.
    """
    y = np.asarray(y).astype(int)  # Ensure integer dtype for sklearn CV

    preprocessor = build_preprocessor()
    pipeline = build_model_pipeline(model, preprocessor)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=False,
        n_jobs=-1,
    )

    # Format results: mean ± std for each metric
    results = {}
    for metric_name, scores in cv_results.items():
        if metric_name.startswith("test_"):
            metric = metric_name[5:]  # remove 'test_' prefix
            results[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
    return results


def run_baseline_comparison() -> pd.DataFrame:
    """Run cross-validation for all baseline models on training data.

    Returns:
        DataFrame with model comparison results (mean ± std for each metric).
    """
    # Load training data
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    models = get_baseline_models()
    results = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        cv_results = evaluate_model_cv(model, X_train, y_train)

        row = {"Model": name}
        for metric, stats in cv_results.items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        results.append(row)

    df_results = pd.DataFrame(results)
    return df_results


def format_results_table(df_results: pd.DataFrame) -> str:
    """Format results as a readable table."""
    lines = []
    lines.append("| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |")
    lines.append("| ----- | ------: | -----: | -------: | --------: | -----: | -: |")

    for _, row in df_results.iterrows():
        line = (
            f"| {row['Model']} | "
            f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f} | "
            f"{row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f} | "
            f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"{row['precision_mean']:.4f} ± {row['precision_std']:.4f} | "
            f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f} | "
            f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f} |"
        )
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    df_results = run_baseline_comparison()
    print("\nBaseline Model Comparison (5-fold Stratified CV on Training Data):")
    print(format_results_table(df_results))

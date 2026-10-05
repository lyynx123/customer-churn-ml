"""Hyperparameter tuning module for Customer Churn Prediction.

This module performs controlled, leakage-safe hyperparameter tuning
using only training data and cross-validation.
"""

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from .config import RANDOM_SEED
from .features import TARGET_COL, build_preprocessor, load_data
from .models import SCORING, build_model_pipeline


def get_lr_search_space() -> dict[str, list]:
    """Return search space for LogisticRegression."""
    return {
        "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "model__solver": ["lbfgs", "liblinear"],
        "model__class_weight": ["balanced", None],
        "model__max_iter": [2000, 5000],
    }


def get_rf_search_space() -> dict[str, list]:
    """Return search space for RandomForestClassifier."""
    return {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 10, 20, 30, 50],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": ["balanced", "balanced_subsample", None],
    }


def create_search(
    model: Any,
    param_distributions: dict[str, list],
    n_iter: int = 30,
    cv_splits: int = 5,
    random_state: int = RANDOM_SEED,
    scoring: str = "roc_auc",
    n_jobs: int = 2,  # Conservative for 8GB RAM
) -> RandomizedSearchCV:
    """Create a RandomizedSearchCV with preprocessing pipeline."""
    preprocessor = build_preprocessor()
    pipeline = build_model_pipeline(model, preprocessor)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=False,  # We don't need to refit on full data yet
        return_train_score=False,
        verbose=0,
    )
    return search


def run_tuning(
    model: Any,
    param_distributions: dict[str, list],
    X: pd.DataFrame,
    y,
    n_iter: int = 30,
    cv_splits: int = 5,
    random_state: int = RANDOM_SEED,
    scoring: str = "roc_auc",
    n_jobs: int = 2,
) -> dict[str, Any]:
    """Run hyperparameter tuning and return results with all metrics."""
    y = np.asarray(y).astype(int)

    search = create_search(
        model=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv_splits=cv_splits,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    search.fit(X, y)

    # Get best estimator and its parameters
    best_params = search.best_params_
    best_score = float(search.best_score_)
    best_idx = search.best_index_

    # Now evaluate the best model with ALL metrics using cross_validate
    best_model = model.__class__(
        **{
            k.replace("model__", ""): v
            for k, v in best_params.items()
            if k.startswith("model__")
        }
    )
    best_model.random_state = random_state

    preprocessor = build_preprocessor()
    pipeline = build_model_pipeline(best_model, preprocessor)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=False,
        n_jobs=n_jobs,
    )

    # Format results: mean ± std for each metric
    metrics = {}
    for metric_name, scores in cv_results.items():
        if metric_name.startswith("test_"):
            metric = metric_name[5:]
            metrics[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }

    result = {
        "best_params": best_params,
        "best_score": best_score,
        "best_index": int(best_idx),
        "n_iter": n_iter,
        "scoring": scoring,
        **metrics,
    }

    return result


def run_all_tuning(
    n_iter_lr: int = 30,
    n_iter_rf: int = 30,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = 2,
) -> dict[str, Any]:
    """Run tuning for all candidate models."""
    # Load training data only
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    results = {}

    # LogisticRegression
    print("Tuning LogisticRegression...")
    lr_model = LogisticRegression(random_state=RANDOM_SEED)
    lr_results = run_tuning(
        model=lr_model,
        param_distributions=get_lr_search_space(),
        X=X_train,
        y=y_train,
        n_iter=n_iter_lr,
        cv_splits=cv_splits,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    results["LogisticRegression"] = lr_results

    # RandomForestClassifier
    print("Tuning RandomForestClassifier...")
    rf_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1)
    rf_results = run_tuning(
        model=rf_model,
        param_distributions=get_rf_search_space(),
        X=X_train,
        y=y_train,
        n_iter=n_iter_rf,
        cv_splits=cv_splits,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    results["RandomForestClassifier"] = rf_results

    return results


def format_tuning_results(results: dict[str, Any]) -> str:
    """Format tuning results as a comparison table."""
    lines = []
    lines.append(
        "| Model | Configuration | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |"
    )
    lines.append(
        "| ----- | ------------- | ------: | -----: | -------: | --------: | -----: | -: |"
    )

    # Baseline from Phase 6 (these are the baseline CV results)
    baselines = {
        "LogisticRegression": {
            "roc_auc": 0.8475,
            "roc_auc_std": 0.0114,
            "pr_auc": 0.6535,
            "pr_auc_std": 0.0320,
            "accuracy": 0.7473,
            "accuracy_std": 0.0088,
            "precision": 0.5156,
            "precision_std": 0.0112,
            "recall": 0.8100,
            "recall_std": 0.0176,
            "f1": 0.6299,
            "f1_std": 0.0088,
        },
        "RandomForestClassifier": {
            "roc_auc": 0.8230,
            "roc_auc_std": 0.0024,
            "pr_auc": 0.6114,
            "pr_auc_std": 0.0147,
            "accuracy": 0.7719,
            "accuracy_std": 0.0089,
            "precision": 0.5611,
            "precision_std": 0.0179,
            "recall": 0.6539,
            "recall_std": 0.0232,
            "f1": 0.6035,
            "f1_std": 0.0113,
        },
    }

    for model_name in ["LogisticRegression", "RandomForestClassifier"]:
        # Baseline row
        base = baselines[model_name]
        lines.append(
            f"| {model_name} | baseline | "
            f"{base['roc_auc']:.4f} ± {base['roc_auc_std']:.4f} | "
            f"{base['pr_auc']:.4f} ± {base['pr_auc_std']:.4f} | "
            f"{base['accuracy']:.4f} ± {base['accuracy_std']:.4f} | "
            f"{base['precision']:.4f} ± {base['precision_std']:.4f} | "
            f"{base['recall']:.4f} ± {base['recall_std']:.4f} | "
            f"{base['f1']:.4f} ± {base['f1_std']:.4f} |"
        )

        # Tuned row
        if model_name in results:
            r = results[model_name]

            # Metrics are nested dicts with 'mean' and 'std'
            def get_metric(metric_name: str, r=r) -> str:
                m = r.get(metric_name, {})
                mean = m.get("mean", 0) if isinstance(m, dict) else 0
                std = m.get("std", 0) if isinstance(m, dict) else 0
                return f"{mean:.4f} ± {std:.4f}"

            lines.append(
                f"| {model_name} | tuned | "
                f"{get_metric('roc_auc')} | "
                f"{get_metric('pr_auc')} | "
                f"{get_metric('accuracy')} | "
                f"{get_metric('precision')} | "
                f"{get_metric('recall')} | "
                f"{get_metric('f1')} |"
            )

    return "\n".join(lines)


def save_tuning_results(results: dict[str, Any], output_path: str):
    """Save tuning results to JSON."""
    # Make serializable
    serializable = {}
    for model_name, result in results.items():
        serializable[model_name] = {
            "best_params": result.get("best_params", {}),
            "best_score": result.get("best_score", 0),
            "best_index": result.get("best_index", 0),
            "n_iter": result.get("n_iter", 0),
            "scoring": result.get("scoring", ""),
            "metrics": {
                metric: {"mean": v.get("mean", 0), "std": v.get("std", 0)}
                for metric, v in result.items()
                if isinstance(v, dict) and "mean" in v and "std" in v
            },
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("Running hyperparameter tuning (training data only)...")
    print("=" * 80)

    results = run_all_tuning(
        n_iter_lr=30,
        n_iter_rf=30,
        cv_splits=5,
        scoring="roc_auc",
        n_jobs=2,
    )

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING RESULTS (5-fold Stratified CV on Training Data)")
    print("=" * 80)
    print(format_tuning_results(results))
    print()

    for name, result in results.items():
        print(f"{name} best params: {result['best_params']}")
        print(f"{name} best ROC-AUC: {result['best_score']:.4f}")
        print()

    # Save artifacts
    output_path = "/home/ahmad_izzuddin_ulinnuha/projects/Customer_Churn/models/tuning_results.json"
    save_tuning_results(results, output_path)
    print(f"Tuning results saved to {output_path}")

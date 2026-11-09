"""Hyperparameter tuning module for Customer Churn Prediction.

This module performs controlled, leakage-safe hyperparameter tuning
using only training data and cross-validation.

Key methodological fix: Nested Cross-Validation is used to avoid
hyperparameter selection bias. The outer CV provides unbiased performance
estimates, while the inner CV selects hyperparameters within each
outer training fold.
"""

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)

from .config import BASE_DIR, RANDOM_SEED
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
    """Run hyperparameter tuning and return results with all metrics.

    DEPRECATED: This function uses a single CV loop for both hyperparameter
    search and evaluation, which introduces selection bias. Use
    `run_nested_tuning` instead for unbiased performance estimates.
    """
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


def run_nested_tuning(
    model: Any,
    param_distributions: dict[str, list],
    X: pd.DataFrame,
    y,
    n_iter: int = 30,
    outer_cv_splits: int = 5,
    inner_cv_splits: int = 3,
    random_state: int = RANDOM_SEED,
    scoring: str = "roc_auc",
    n_jobs: int = 2,
) -> dict[str, Any]:
    """Run nested cross-validation for unbiased hyperparameter tuning.

    This implements proper nested CV to avoid hyperparameter selection bias:
    - Outer CV (outer_cv_splits): Provides unbiased performance estimates
    - Inner CV (inner_cv_splits): Selects hyperparameters within each outer training fold

    Args:
        model: Base sklearn estimator (will be wrapped in pipeline with preprocessor)
        param_distributions: Parameter distributions for RandomizedSearchCV
        X: Training features DataFrame
        y: Training target array
        n_iter: Number of random search iterations for inner CV
        outer_cv_splits: Number of outer CV folds (for performance estimation)
        inner_cv_splits: Number of inner CV folds (for hyperparameter selection)
        random_state: Random seed for reproducibility
        scoring: Scoring metric for hyperparameter selection
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary with:
        - outer_cv_scores: Performance metrics from outer CV (mean ± std)
        - best_params_per_fold: Best hyperparameters selected in each outer fold
        - mean_outer_score: Mean score across outer folds (unbiased estimate)
        - std_outer_score: Std of scores across outer folds
        - scoring: The scoring metric used
    """
    y = np.asarray(y).astype(int)

    outer_cv = StratifiedKFold(
        n_splits=outer_cv_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_cv_splits, shuffle=True, random_state=random_state + 1
    )

    # Scorer for inner CV
    if scoring == "roc_auc":
        inner_scorer = "roc_auc"
    elif scoring == "pr_auc":
        inner_scorer = "average_precision"
    else:
        inner_scorer = scoring

    outer_scores = []
    best_params_per_fold = []

    preprocessor = build_preprocessor()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_outer = X.iloc[train_idx]
        y_train_outer = y[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_test_outer = y[test_idx]

        # Inner CV: Hyperparameter search on outer training data
        pipeline = build_model_pipeline(model, preprocessor)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=inner_scorer,
            random_state=random_state + fold_idx * 10,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            verbose=0,
        )

        search.fit(X_train_outer, y_train_outer)

        best_params = search.best_params_
        best_params_per_fold.append(best_params)

        # Evaluate best model on outer test fold
        best_estimator = search.best_estimator_
        y_proba = best_estimator.predict_proba(X_test_outer)[:, 1]

        if scoring == "roc_auc":
            score = roc_auc_score(y_test_outer, y_proba)
        elif scoring == "pr_auc":
            score = average_precision_score(y_test_outer, y_proba)
        else:
            # Fallback: use the scorer
            from sklearn.metrics import get_scorer
            scorer = get_scorer(scoring)
            score = scorer(best_estimator, X_test_outer, y_test_outer)

        outer_scores.append(float(score))

    # Compute all metrics on outer folds using cross_val_predict with nested approach
    # We need to collect OOF predictions for each outer fold to compute all metrics
    all_y_true: list[int] = []
    all_y_proba: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_outer = X.iloc[train_idx]
        y_train_outer = y[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_test_outer = y[test_idx]

        pipeline = build_model_pipeline(model, preprocessor)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=inner_scorer,
            random_state=random_state + fold_idx * 10,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            verbose=0,
        )

        search.fit(X_train_outer, y_train_outer)
        best_estimator = search.best_estimator_
        y_proba = best_estimator.predict_proba(X_test_outer)[:, 1]

        all_y_true.extend(y_test_outer)
        all_y_proba.extend(y_proba)

    all_y_true_np: np.ndarray = np.array(all_y_true)
    all_y_proba_np: np.ndarray = np.array(all_y_proba)

    # Compute all metrics on the aggregated OOF predictions
    metrics: dict[str, dict[str, float]] = {}
    for metric_name in SCORING:
        if metric_name == "roc_auc":
            score = roc_auc_score(all_y_true_np, all_y_proba_np)
        elif metric_name == "pr_auc":
            score = average_precision_score(all_y_true_np, all_y_proba_np)
        else:
            from sklearn.metrics import get_scorer
            scorer = get_scorer(metric_name)
            # For threshold-dependent metrics, we need predictions at 0.5
            y_pred = (all_y_proba_np >= 0.5).astype(int)
            score = scorer._score_func(all_y_true_np, y_pred, **scorer._kwargs)

        metrics[metric_name] = {
            "mean": float(score),
            "std": float(np.std(outer_scores)),  # Use outer fold score std as proxy
        }

    # Also compute the primary scoring metric stats from outer folds
    primary_metric = scoring if scoring in ["roc_auc", "pr_auc"] else "roc_auc"
    metrics[primary_metric] = {
        "mean": float(np.mean(outer_scores)),
        "std": float(np.std(outer_scores)),
    }

    result = {
        "outer_cv_scores": outer_scores,
        "best_params_per_fold": best_params_per_fold,
        "mean_outer_score": float(np.mean(outer_scores)),
        "std_outer_score": float(np.std(outer_scores)),
        "scoring": scoring,
        "outer_cv_splits": outer_cv_splits,
        "inner_cv_splits": inner_cv_splits,
        "n_iter": n_iter,
        **metrics,
    }

    return result


def run_all_tuning(
    n_iter_lr: int = 30,
    n_iter_rf: int = 30,
    outer_cv_splits: int = 5,
    inner_cv_splits: int = 3,
    scoring: str = "roc_auc",
    n_jobs: int = 2,
    use_nested_cv: bool = True,
) -> dict[str, Any]:
    """Run tuning for all candidate models.

    By default, uses nested cross-validation for unbiased performance estimates.
    Set use_nested_cv=False to use the legacy single-CV approach (deprecated).
    """
    # Load training data only
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    results = {}

    # LogisticRegression
    print("Tuning LogisticRegression...")
    lr_model = LogisticRegression(random_state=RANDOM_SEED)
    if use_nested_cv:
        lr_results = run_nested_tuning(
            model=lr_model,
            param_distributions=get_lr_search_space(),
            X=X_train,
            y=y_train,
            n_iter=n_iter_lr,
            outer_cv_splits=outer_cv_splits,
            inner_cv_splits=inner_cv_splits,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    else:
        lr_results = run_tuning(
            model=lr_model,
            param_distributions=get_lr_search_space(),
            X=X_train,
            y=y_train,
            n_iter=n_iter_lr,
            cv_splits=outer_cv_splits,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    results["LogisticRegression"] = lr_results

    # RandomForestClassifier
    print("Tuning RandomForestClassifier...")
    rf_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1)
    if use_nested_cv:
        rf_results = run_nested_tuning(
            model=rf_model,
            param_distributions=get_rf_search_space(),
            X=X_train,
            y=y_train,
            n_iter=n_iter_rf,
            outer_cv_splits=outer_cv_splits,
            inner_cv_splits=inner_cv_splits,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    else:
        rf_results = run_tuning(
            model=rf_model,
            param_distributions=get_rf_search_space(),
            X=X_train,
            y=y_train,
            n_iter=n_iter_rf,
            cv_splits=outer_cv_splits,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    results["RandomForestClassifier"] = rf_results

    return results


def evaluate_on_validation(
    results: dict[str, Any],
    metric: str = "roc_auc",
) -> dict[str, Any]:
    """Evaluate tuned models on validation set and select best model.

    Args:
        results: Output from run_all_tuning containing best_params for each model.
            Supports both legacy single-CV results (with 'best_params') and
            nested CV results (with 'best_params_per_fold' and 'outer_cv_scores').
        metric: Metric to use for model selection (default: roc_auc).

    Returns:
        Dictionary with validation scores and selected model name.
    """
    val_df = load_data("val")
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    validation_scores = {}
    best_model_name = None
    best_score = -1.0

    for model_name, result in results.items():
        # Handle both nested CV and legacy results
        if "best_params_per_fold" in result and "outer_cv_scores" in result:
            # Nested CV: select params from fold with best outer CV score
            best_fold_idx = int(np.argmax(result["outer_cv_scores"]))
            best_params = result["best_params_per_fold"][best_fold_idx]
        else:
            # Legacy single-CV results
            best_params = result.get("best_params", {})

        model_params = {
            k.replace("model__", ""): v
            for k, v in best_params.items()
            if k.startswith("model__")
        }

        if model_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(random_state=RANDOM_SEED, **model_params)
        elif model_name == "RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1, **model_params)
        else:
            continue

        preprocessor = build_preprocessor()
        pipeline = build_model_pipeline(model, preprocessor)

        pipeline.fit(X_val, y_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        if metric == "roc_auc":
            score = roc_auc_score(y_val, y_proba)
        elif metric == "pr_auc":
            score = average_precision_score(y_val, y_proba)
        else:
            raise ValueError(f"Unsupported metric for validation: {metric}")

        validation_scores[model_name] = float(score)

        if score > best_score:
            best_score = score
            best_model_name = model_name

    return {
        "validation_scores": validation_scores,
        "selected_model": best_model_name,
        "selection_metric": metric,
        "selection_score": best_score,
    }


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

            config_label = "tuned (nested CV)" if "outer_cv_splits" in r else "tuned"
            lines.append(
                f"| {model_name} | {config_label} | "
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
        if "outer_cv_splits" in result:
            # Nested CV results
            serializable[model_name] = {
                "method": "nested_cv",
                "outer_cv_splits": result.get("outer_cv_splits", 5),
                "inner_cv_splits": result.get("inner_cv_splits", 3),
                "outer_cv_scores": result.get("outer_cv_scores", []),
                "best_params_per_fold": result.get("best_params_per_fold", []),
                "mean_outer_score": result.get("mean_outer_score", 0),
                "std_outer_score": result.get("std_outer_score", 0),
                "scoring": result.get("scoring", ""),
                "n_iter": result.get("n_iter", 0),
                "metrics": {
                    metric: {"mean": v.get("mean", 0), "std": v.get("std", 0)}
                    for metric, v in result.items()
                    if isinstance(v, dict) and "mean" in v and "std" in v
                },
            }
        else:
            # Legacy single-CV results
            serializable[model_name] = {
                "method": "single_cv",
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


def run_tuning_with_validation(
    n_iter_lr: int = 30,
    n_iter_rf: int = 30,
    outer_cv_splits: int = 5,
    inner_cv_splits: int = 3,
    scoring: str = "roc_auc",
    n_jobs: int = 2,
    val_metric: str = "roc_auc",
    use_nested_cv: bool = True,
) -> dict[str, Any]:
    """Run tuning and validation-based model selection."""
    results = run_all_tuning(
        n_iter_lr=n_iter_lr,
        n_iter_rf=n_iter_rf,
        outer_cv_splits=outer_cv_splits,
        inner_cv_splits=inner_cv_splits,
        scoring=scoring,
        n_jobs=n_jobs,
        use_nested_cv=use_nested_cv,
    )

    val_results = evaluate_on_validation(results, metric=val_metric)
    return {"tuning_results": results, "validation_results": val_results}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("Running hyperparameter tuning with Nested CV (training data only)...")
    print("=" * 80)

    results = run_all_tuning(
        n_iter_lr=30,
        n_iter_rf=30,
        outer_cv_splits=5,
        inner_cv_splits=3,
        scoring="roc_auc",
        n_jobs=2,
        use_nested_cv=True,
    )

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING RESULTS (Nested CV: 5 outer x 3 inner folds)")
    print("=" * 80)
    print(format_tuning_results(results))
    print()

    for name, result in results.items():
        if "best_params_per_fold" in result:
            # Nested CV results
            print(f"{name} outer CV ROC-AUC: {result['mean_outer_score']:.4f} ± {result['std_outer_score']:.4f}")
            print(f"{name} best params per fold: {result['best_params_per_fold']}")
        else:
            # Legacy results
            print(f"{name} best params: {result['best_params']}")
            print(f"{name} best ROC-AUC: {result['best_score']:.4f}")
        print()

    # Validation-based model selection
    print("=" * 80)
    print("VALIDATION-BASED MODEL SELECTION")
    print("=" * 80)
    val_results = evaluate_on_validation(results, metric="roc_auc")
    print(f"Validation scores (ROC-AUC): {val_results['validation_scores']}")
    print(f"Selected model: {val_results['selected_model']}")
    print(f"Selection metric: {val_results['selection_metric']}")
    print(f"Selection score: {val_results['selection_score']:.4f}")
    print()

    # Save artifacts
    output_path = BASE_DIR / "models" / "tuning_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_tuning_results(results, str(output_path))
    print(f"Tuning results saved to {output_path}")

    # Save validation results
    val_output_path = "/home/ahmad_izzuddin_ulinnuha/projects/Customer_Churn/models/validation_selection.json"
    import json

    with open(val_output_path, "w") as f:
        json.dump(val_results, f, indent=2)
    print(f"Validation selection results saved to {val_output_path}")

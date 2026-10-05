"""Tests for Phase 8A hyperparameter tuning."""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from customer_churn.config import RANDOM_SEED
from customer_churn.features import TARGET_COL, build_preprocessor, load_data
from customer_churn.models import build_model_pipeline
from customer_churn.tune import (
    create_search,
    format_tuning_results,
    get_lr_search_space,
    get_rf_search_space,
    run_all_tuning,
    run_tuning,
)


@pytest.fixture(scope="module")
def train_data():
    """Load training data."""
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    return X_train, y_train


def test_lr_search_space():
    """LogisticRegression search space should have expected parameters."""
    space = get_lr_search_space()
    assert "model__C" in space
    assert "model__solver" in space
    assert "model__class_weight" in space
    assert "model__max_iter" in space

    # Check reasonable ranges
    assert all(c > 0 for c in space["model__C"])
    assert "lbfgs" in space["model__solver"]
    assert "liblinear" in space["model__solver"]


def test_rf_search_space():
    """RandomForest search space should have expected parameters."""
    space = get_rf_search_space()
    assert "model__n_estimators" in space
    assert "model__max_depth" in space
    assert "model__min_samples_split" in space
    assert "model__min_samples_leaf" in space
    assert "model__max_features" in space
    assert "model__class_weight" in space


def test_create_search_returns_randomized_search_cv():
    """create_search should return a RandomizedSearchCV with Pipeline."""
    model = LogisticRegression(random_state=RANDOM_SEED)
    search = create_search(
        model=model,
        param_distributions=get_lr_search_space(),
        n_iter=5,
        n_jobs=1,
    )

    from sklearn.model_selection import RandomizedSearchCV

    assert isinstance(search, RandomizedSearchCV)
    assert isinstance(search.estimator, Pipeline)
    assert search.n_iter == 5
    assert search.random_state == RANDOM_SEED

    # Check pipeline steps
    steps = [name for name, _ in search.estimator.steps]
    assert "preprocessor" in steps
    assert "model" in steps


def test_pipeline_contains_preprocessing_and_classifier():
    """Pipeline must contain preprocessing + classifier."""
    # Just check the pipeline structure without fitting
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=RANDOM_SEED)
    preprocessor = build_preprocessor()
    pipeline = build_model_pipeline(model, preprocessor)

    assert isinstance(pipeline, Pipeline)
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names
    assert "model" in step_names


def test_cv_uses_stratified_kfold():
    """CV strategy must be StratifiedKFold with correct params."""
    model = LogisticRegression(random_state=RANDOM_SEED)
    search = create_search(
        model=model,
        param_distributions=get_lr_search_space(),
        n_iter=5,
        cv_splits=5,
        random_state=RANDOM_SEED,
    )

    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.n_splits == 5
    assert search.cv.shuffle is True
    assert search.cv.random_state == RANDOM_SEED


def test_random_state_consistency():
    """random_state=42 must be used consistently."""
    model = LogisticRegression(random_state=RANDOM_SEED)
    search = create_search(
        model=model,
        param_distributions=get_lr_search_space(),
        n_iter=5,
        cv_splits=5,
        random_state=RANDOM_SEED,
    )

    assert search.random_state == RANDOM_SEED
    assert search.cv.random_state == RANDOM_SEED


def test_tuning_does_not_require_test_data():
    """Tuning code should only use training data."""
    # The run_all_tuning function only calls load_data('train')
    # This is verified by checking the source doesn't call load_data('test')
    import inspect

    source = inspect.getsource(run_all_tuning)
    assert "load_data('test')" not in source
    assert 'load_data("test")' not in source


def test_target_excluded_from_x():
    """Target must be excluded from X in tuning."""
    import inspect

    source = inspect.getsource(run_all_tuning)
    # Should drop TARGET_COL from X
    assert "drop(columns=[TARGET_COL])" in source


def test_tuning_returns_expected_metrics():
    """run_tuning should return all expected metrics with mean/std."""
    # Quick smoke test with minimal iterations
    from sklearn.linear_model import LogisticRegression

    from customer_churn.features import TARGET_COL, load_data

    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL]).iloc[:200]
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values[:200]

    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    result = run_tuning(
        model=model,
        param_distributions={"model__C": [0.1, 1.0], "model__solver": ["lbfgs"]},
        X=X_train,
        y=y_train,
        n_iter=2,
        cv_splits=3,
        n_jobs=1,
    )

    # Check all expected metrics present
    expected_metrics = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]
    for metric in expected_metrics:
        assert metric in result, f"Missing metric: {metric}"
        assert isinstance(result[metric], dict), f"{metric} should be dict"
        assert "mean" in result[metric], f"{metric} missing mean"
        assert "std" in result[metric], f"{metric} missing std"
        assert isinstance(result[metric]["mean"], float)
        assert isinstance(result[metric]["std"], float)
        assert 0.0 <= result[metric]["mean"] <= 1.0
        assert result[metric]["std"] >= 0.0


def test_baseline_and_tuned_results_compatible_schema():
    """Baseline and tuned results should have compatible schemas for comparison."""
    # Test the format function with mock data
    mock_results = {
        "LogisticRegression": {
            "roc_auc": {"mean": 0.85, "std": 0.01},
            "pr_auc": {"mean": 0.65, "std": 0.02},
            "accuracy": {"mean": 0.80, "std": 0.01},
            "precision": {"mean": 0.65, "std": 0.02},
            "recall": {"mean": 0.55, "std": 0.03},
            "f1": {"mean": 0.60, "std": 0.02},
        }
    }

    table = format_tuning_results(mock_results)
    assert "LogisticRegression" in table
    assert "baseline" in table
    assert "tuned" in table
    assert "0.8500" in table
    assert "| Model |" in table


def test_tuning_does_not_load_test_data():
    """Tuning should not load test data."""
    # Verify by checking source doesn't reference test data
    import inspect

    source = inspect.getsource(run_all_tuning)
    assert "test" not in source.lower() or "test" in source.lower().split(
        "test_"
    )  # test_data is ok, test.parquet is not
    # More specifically: should not load test parquet
    assert "test.parquet" not in source


def test_best_params_retrievable():
    """Best parameters should be retrievable from tuning result."""
    from sklearn.linear_model import LogisticRegression

    from customer_churn.features import TARGET_COL, load_data

    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL]).iloc[:100]
    y_train = train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values[:100]

    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    result = run_tuning(
        model=model,
        param_distributions={"model__C": [0.1, 1.0], "model__solver": ["lbfgs"]},
        X=X_train,
        y=y_train,
        n_iter=2,
        cv_splits=3,
        n_jobs=1,
    )

    assert "best_params" in result
    assert isinstance(result["best_params"], dict)
    assert "model__C" in result["best_params"]
    assert "model__solver" in result["best_params"]

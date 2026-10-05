import numpy as np
import pytest
from sklearn.compose import ColumnTransformer

from customer_churn.config import PROCESSED_DATA_PATH
from customer_churn.features import (
    NUMERIC_COLS,
    TARGET_COL,
    _get_categorical_columns,
    build_preprocessor,
    load_data,
    preprocess_and_transform,
)
from customer_churn.split import split_data


@pytest.fixture(scope="module")
def splits():
    # Ensure splits exist; create if missing
    train_path = PROCESSED_DATA_PATH.parent / "train.parquet"
    if not train_path.exists():
        split_data()
    train = load_data("train")
    val = load_data("val")
    test = load_data("test")
    return train, val, test


def test_get_categorical_columns():
    cat_cols = _get_categorical_columns()
    assert isinstance(cat_cols, list)
    # Should contain known categorical columns
    expected = [
        "Contract",
        "Dependents",
        "DeviceProtection",
        "InternetService",
        "MultipleLines",
        "OnlineBackup",
        "OnlineSecurity",
        "PaperlessBilling",
        "Partner",
        "PaymentMethod",
        "PhoneService",
        "StreamingMovies",
        "StreamingTV",
        "TechSupport",
        "gender",
    ]
    for col in expected:
        assert col in cat_cols
    # No numeric or target
    for col in NUMERIC_COLS + [TARGET_COL]:
        assert col not in cat_cols


def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    # Transformers should have expected names
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert set(transformer_names) == {"num", "cat"}


def test_preprocess_fit_transform(splits):
    train, val, test = splits
    preprocessor = build_preprocessor()

    # Fit on train
    train_trans = preprocess_and_transform("train", preprocessor, fit=True)
    assert train_trans.shape[0] == train.shape[0]
    assert TARGET_COL in train_trans.columns
    assert train_trans.isna().sum().sum() == 0

    # Transform val
    val_trans = preprocess_and_transform("val", preprocessor, fit=False)
    assert val_trans.shape[0] == val.shape[0]
    assert val_trans.columns.tolist() == train_trans.columns.tolist()
    assert val_trans.isna().sum().sum() == 0

    # Transform test
    test_trans = preprocess_and_transform("test", preprocessor, fit=False)
    assert test_trans.shape[0] == test.shape[0]
    assert test_trans.columns.tolist() == train_trans.columns.tolist()
    assert test_trans.isna().sum().sum() == 0

    # Feature count same across splits
    assert train_trans.shape[1] == val_trans.shape[1] == test_trans.shape[1]


def test_unseen_categorical_handling(splits):
    """Ensure preprocessor handles unseen categorical values gracefully."""
    train, _, _ = splits
    preprocessor = build_preprocessor()
    preprocessor.fit(train.drop(columns=[TARGET_COL]))

    # Create a single row with a new category for a known column
    new_row = train.drop(columns=[TARGET_COL]).iloc[:1].copy()
    # Pick a categorical column
    cat_cols = [c for c in new_row.columns if c not in NUMERIC_COLS]
    col = cat_cols[0]
    new_row[col] = "UNSEEN_VALUE"
    # Should not raise
    transformed = preprocessor.transform(new_row)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_target_exclusion():
    """Ensure target column is not present in feature matrix passed to preprocessor."""
    train = load_data("train")
    preprocessor = build_preprocessor()
    X = train.drop(columns=[TARGET_COL])
    # Fit
    preprocessor.fit(X)
    # Check that target column not in feature names
    feature_names = preprocessor.get_feature_names_out()
    assert TARGET_COL not in feature_names


def test_no_leakage_across_splits(splits):
    """Validation and test transforms must not fit."""
    train, val, test = splits
    preprocessor = build_preprocessor()
    preprocessor.fit(train.drop(columns=[TARGET_COL]))

    # Ensure val/test only transformed, not re-fitted
    val_trans = preprocess_and_transform("val", preprocessor, fit=False)
    test_trans = preprocess_and_transform("test", preprocessor, fit=False)

    # Feature names identical
    assert val_trans.columns.tolist() == test_trans.columns.tolist()
    # No additional fitting: shapes consistent with original rows
    assert val_trans.shape[0] == val.shape[0]
    assert test_trans.shape[0] == test.shape[0]

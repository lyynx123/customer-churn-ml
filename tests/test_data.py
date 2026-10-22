"""Regression tests for categorical missing-value handling in data.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from customer_churn.data import clean_data
from customer_churn.features import build_preprocessor


def test_none_categorical_preserved_as_na():
    """None in categorical column should be preserved as pd.NA."""
    df = pd.DataFrame({"cat_col": ["a", None, "b"], "TotalCharges": [1.0, 2.0, 3.0]})
    result = clean_data(df)
    assert pd.isna(result.loc[1, "cat_col"])


def test_nan_categorical_preserved_as_na():
    """np.nan in categorical column should be preserved as pd.NA."""
    df = pd.DataFrame({"cat_col": ["a", np.nan, "b"], "TotalCharges": [1.0, 2.0, 3.0]})
    result = clean_data(df)
    assert pd.isna(result.loc[1, "cat_col"])


def test_pdna_categorical_preserved_as_na():
    """pd.NA in categorical column should be preserved as pd.NA."""
    df = pd.DataFrame({"cat_col": ["a", pd.NA, "b"], "TotalCharges": [1.0, 2.0, 3.0]})
    result = clean_data(df)
    assert pd.isna(result.loc[1, "cat_col"])


def test_whitespace_only_becomes_na():
    """Whitespace-only strings should be treated as missing."""
    df = pd.DataFrame({"cat_col": ["a", "   ", "b"], "TotalCharges": [1.0, 2.0, 3.0]})
    result = clean_data(df)
    assert pd.isna(result.loc[1, "cat_col"])


def test_ordinary_strings_stripped():
    """Ordinary strings should be stripped of whitespace."""
    df = pd.DataFrame(
        {"cat_col": ["  a  ", "b", "c  "], "TotalCharges": [1.0, 2.0, 3.0]}
    )
    result = clean_data(df)
    assert list(result["cat_col"]) == ["a", "b", "c"]


def test_literal_nan_category_preserved():
    """Legitimate literal 'nan' category should be preserved."""
    df = pd.DataFrame({"cat_col": ["a", "nan", "b"], "TotalCharges": [1.0, 2.0, 3.0]})
    result = clean_data(df)
    assert result.loc[1, "cat_col"] == "nan"


def test_imputer_compatibility():
    """End-to-end compatibility with SimpleImputer(strategy='most_frequent')."""
    df = pd.DataFrame(
        {"cat_col": ["a", pd.NA, "a", "b"], "TotalCharges": [1.0, 2.0, 3.0, 4.0]}
    )
    result = clean_data(df)
    imputer = SimpleImputer(strategy="most_frequent")
    imputed = imputer.fit_transform(result[["cat_col"]])
    # Most frequent is "a", so missing should become "a"
    assert imputed[1, 0] == "a"


def test_preprocessor_pipeline_compatibility():
    """Full preprocessing pipeline should handle cleaned data with missing categoricals."""
    df = pd.DataFrame(
        {
            "cat_col": ["a", pd.NA, "a", "b"],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
            "SeniorCitizen": [0, 1, 0, 1],
            "TotalCharges": [100.0, 200.0, 300.0, 400.0],
            "tenure": [1, 2, 3, 4],
        }
    )
    result = clean_data(df)

    # Build preprocessor and fit
    preprocessor = build_preprocessor(cat_cols=["cat_col"])
    has_churn = "Churn" in result.columns
    X = result.drop(columns=["Churn"]) if has_churn else result
    _ = result["Churn"].replace({"No": 0, "Yes": 1}) if has_churn else None

    # This should not raise an error
    X_trans = preprocessor.fit_transform(X)
    assert X_trans.shape[0] == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

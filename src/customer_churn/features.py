import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import PROCESSED_DATA_PATH

# Explicit numeric feature list – determined from the processed dataset
NUMERIC_COLS = ["MonthlyCharges", "SeniorCitizen", "TotalCharges", "tenure"]
TARGET_COL = "Churn"


def _get_categorical_columns() -> list:
    """Return categorical column names from the processed parquet file.
    Reads only the column schema (no data) using pyarrow for efficiency.
    """
    import pyarrow.parquet as pq

    table = pq.read_schema(PROCESSED_DATA_PATH)
    all_cols = [field.name for field in table]
    # Exclude numeric and target
    cat = [c for c in all_cols if c not in NUMERIC_COLS + [TARGET_COL]]
    return cat


def load_data(split: str) -> pd.DataFrame:
    """Load a processed split parquet file (train/val/test)."""
    path = PROCESSED_DATA_PATH.parent / f"{split}.parquet"
    return pd.read_parquet(path)


def build_preprocessor(cat_cols: list | None = None) -> ColumnTransformer:
    """Construct a ColumnTransformer for numeric and categorical features.
    If ``cat_cols`` is ``None`` the function determines them via ``_get_categorical_columns``.
    """
    if cat_cols is None:
        cat_cols = _get_categorical_columns()
    # Numeric pipeline – median imputation (no scaling needed for tree‑based models)
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    # Categorical pipeline – most‑frequent imputation + one‑hot encoding (dense output)
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return preprocessor


def preprocess_and_transform(
    split: str, preprocessor: ColumnTransformer, fit: bool = False
) -> pd.DataFrame:
    """Transform a split using ``preprocessor``.
    If ``fit`` is ``True`` the preprocessor is fitted on the data (used for training split).
    Returns a DataFrame with transformed features and the target column.
    """
    df = load_data(split)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].replace({"No": 0, "Yes": 1})
    if fit:
        X_trans = preprocessor.fit_transform(X)
    else:
        X_trans = preprocessor.transform(X)
    # Feature names – works for dense output
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(np.asarray(X_trans), columns=feature_names)
    X_df[TARGET_COL] = y.values
    return X_df


if __name__ == "__main__":
    # Example usage – fit on train, transform all splits and report shapes
    preproc = build_preprocessor()
    for sp, do_fit in [("train", True), ("val", False), ("test", False)]:
        transformed = preprocess_and_transform(sp, preproc, fit=do_fit)
        print(f"{sp.title()} shape: {transformed.shape}")
        print(transformed.head())

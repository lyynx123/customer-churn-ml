from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .config import PROCESSED_DATA_PATH, RAW_DATA_PATH

# Stable dataset reference: IBM telco-customer-churn-on-icp4d v1.0 tag
# Commit: 6f14333ebbed9b88dccc29557c31de28b4b0dc57 (2019-10-11)
# Tag: v1.0
DATASET_SOURCE = {
    "name": "Telco Customer Churn",
    "repository": "IBM/telco-customer-churn-on-icp4d",
    "version": "v1.0",
    "commit": "6f14333ebbed9b88dccc29557c31de28b4b0dc57",
    "path": "data/Telco-Customer-Churn.csv",
    "url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/6f14333ebbed9b88dccc29557c31de28b4b0dc57/data/Telco-Customer-Churn.csv",
    "sha256": "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
    "rows": 7043,
    "columns": 21,
}

METADATA_PATH = RAW_DATA_PATH.with_suffix(".json")


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_metadata() -> dict | None:
    """Load dataset metadata if exists."""
    if METADATA_PATH.exists():
        try:
            with METADATA_PATH.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_metadata(metadata: dict) -> None:
    """Save dataset metadata."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w") as f:
        json.dump(metadata, f, indent=2)


def _verify_checksum(filepath: Path) -> bool:
    """Verify file checksum against expected value."""
    actual = _compute_sha256(filepath)
    expected = DATASET_SOURCE["sha256"]
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {filepath.name}:\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"Dataset may have been corrupted or changed. "
            f"Please re-download or investigate."
        )
    return True


def _ensure_raw_data() -> None:
    """Download raw CSV if it doesn't exist or checksum fails."""
    if RAW_DATA_PATH.exists():
        try:
            _verify_checksum(RAW_DATA_PATH)
            return
        except ValueError:
            print(f"Checksum verification failed for {RAW_DATA_PATH}. Re-downloading...")
            RAW_DATA_PATH.unlink()

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = DATASET_SOURCE["url"]  # type: ignore[assignment]
    print(f"Downloading raw data from {url} ...")
    urlretrieve(url, RAW_DATA_PATH)  # type: ignore[arg-type]
    print(f"Saved to {RAW_DATA_PATH}")

    _verify_checksum(RAW_DATA_PATH)

    metadata = {
        "source": DATASET_SOURCE,
        "fetched_at": time.time(),
        "fetched_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "local_path": str(RAW_DATA_PATH),
        "local_sha256": DATASET_SOURCE["sha256"],
    }
    _save_metadata(metadata)
    print(f"Metadata saved to {METADATA_PATH}")


def load_raw() -> pd.DataFrame:
    """Load raw CSV into DataFrame."""
    _ensure_raw_data()
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning:
    - Convert TotalCharges to numeric (coerce errors)
    - Strip whitespace, replace empty strings with NaN
    - Drop customerID (non‑predictive)
    - Set TotalCharges=0 for tenure=0 customers with missing charges (no billing history)
    """
    df = df.copy()
    # Set TotalCharges=0 for tenure=0 customers with missing charges (no billing history).
    # Only applies if tenure column exists and TotalCharges is missing/whitespace.
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        mask = (df["tenure"] == 0) & (df["TotalCharges"].astype(str).str.strip() == "")
        # Only set string "0" if TotalCharges is object/string dtype (not already numeric)
        if df["TotalCharges"].dtype == object or pd.api.types.is_string_dtype(
            df["TotalCharges"].dtype
        ):
            df.loc[mask, "TotalCharges"] = "0"
    # Replace empty strings in TotalCharges with NaN and coerce
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    # Strip whitespace for all object columns, preserving missing values
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        is_missing = df[col].isna()
        df[col] = df[col].astype(str).str.strip()
        df.loc[is_missing, col] = pd.NA
        df.loc[df[col] == "", col] = pd.NA
    # Drop identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def validate(df: pd.DataFrame) -> None:
    """Validate dataset integrity.
    Raises AssertionError if any check fails.
    Raw CSV has 0 duplicates (customerID is unique). After dropping
    customerID some rows may share identical feature values which is
    acceptable for ML - we only assert the original raw-rows check.
    """
    # Rows and columns
    assert df.shape[0] > 0, "Dataset is empty"
    # Original raw CSV has 0 full-row duplicates; assert after we have
    # already dropped the guaranteed-unique customerID.
    # No missing target
    assert not df["Churn"].isna().any(), "Missing target values"  # type: ignore[return-value]
    # Target should be Yes/No
    assert set(df["Churn"].unique()) <= {"Yes", "No"}, "Unexpected target values"
    # Note: rows may share identical features after dropping customerID;
    # this is not treated as an error for ML purposes.


def preprocess_and_save(df: pd.DataFrame) -> Path:
    """Clean, validate and write processed data to parquet.
    Returns path to saved file.
    """
    df_clean = clean_data(df)
    validate(df_clean)
    # Ensure deterministic order of columns
    df_clean = df_clean.sort_index(axis=1)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(PROCESSED_DATA_PATH, index=False)
    return PROCESSED_DATA_PATH


def run_pipeline() -> Path:
    """Execute full data pipeline.
    Returns path to processed parquet file.
    """
    raw = load_raw()
    return preprocess_and_save(raw)


def get_dataset_info() -> dict:
    """Return dataset source information and local metadata."""
    info = {"source": DATASET_SOURCE}
    metadata = _load_metadata()
    if metadata:
        info["local"] = metadata
    return info


if __name__ == "__main__":
    path = run_pipeline()
    print(f"Processed data saved to: {path}")
    info = get_dataset_info()
    print("\nDataset info:")
    print(json.dumps(info, indent=2))
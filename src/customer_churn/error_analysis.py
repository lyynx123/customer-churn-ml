"""Phase 10 - Error analysis on frozen held-out test predictions.

Descriptive analysis only. No retraining, no tuning, no threshold search.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_PATH
from .features import NUMERIC_COLS, TARGET_COL
from .predict import DEFAULT_THRESHOLD, load_pipeline

# Probability below which a false negative is "high confidence"
HIGH_CONF_PROB = 0.20
# Probability above which a false positive is "high confidence"
HIGH_CONF_PROB_FP = 0.80
# Borderline band around the decision threshold
BORDERLINE_BAND = 0.10

ERROR_TYPES = ("TN", "FP", "FN", "TP")


def classify_errors(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> np.ndarray:
    """Return per-sample error type labels in ERROR_TYPES order."""
    y_pred = (y_proba >= threshold).astype(int)
    error_type = np.full(len(y_true), "", dtype=object)
    error_type[(y_true == 0) & (y_pred == 0)] = "TN"
    error_type[(y_true == 0) & (y_pred == 1)] = "FP"
    error_type[(y_true == 1) & (y_pred == 0)] = "FN"
    error_type[(y_true == 1) & (y_pred == 1)] = "TP"
    return error_type


def error_counts(error_type: np.ndarray) -> dict[str, int]:
    """Count occurrences of each error type."""
    values, counts = np.unique(error_type, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when the denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


def error_rates(counts: dict[str, int]) -> dict[str, float]:
    """Derive FPR, FNR and positive prediction rate from error counts."""
    tn, fp, fn, tp = counts["TN"], counts["FP"], counts["FN"], counts["TP"]
    total = tn + fp + fn + tp
    return {
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_negative_rate": safe_divide(fn, fn + tp),
        "positive_prediction_rate": safe_divide(tp + fp, total),
    }


def load_frozen_predictions(threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Load the serialized pipeline and score the frozen test split.

    The pipeline was fitted during Phase 9 on training data only. No fitting,
    tuning or threshold search happens here.
    """
    pipeline = load_pipeline()
    test_df = pd.read_parquet(PROCESSED_DATA_PATH.parent / "test.parquet")
    y_true = test_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).to_numpy()
    y_proba = pipeline.predict_proba(test_df.drop(columns=[TARGET_COL]))[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "features": test_df.drop(columns=[TARGET_COL]),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "error_type": classify_errors(y_true, y_proba, threshold),
    }


def subgroup_error_rates(
    features: pd.DataFrame, error_type: np.ndarray, column: str
) -> pd.DataFrame:
    """Per-value error counts and rates for one feature column.

    Small groups are reported with their counts so they are not over-read.
    """
    frame = pd.DataFrame({"value": features[column].to_numpy(), "error": error_type})
    grouped = (
        frame.groupby("value", observed=True)["error"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for key in ERROR_TYPES:
        if key not in grouped.columns:
            grouped[key] = 0
    grouped = grouped[list(ERROR_TYPES)]
    grouped["total"] = grouped.sum(axis=1)
    grouped["fp_rate"] = grouped["FP"] / (grouped["FP"] + grouped["TN"])
    grouped["fn_rate"] = grouped["FN"] / (grouped["FN"] + grouped["TP"])
    return grouped.reset_index()


def numerical_error_summary(
    features: pd.DataFrame, error_type: np.ndarray, column: str
) -> dict[str, Any]:
    """Mean and median of one numeric feature per error type."""
    summary: dict[str, Any] = {}
    for key in ERROR_TYPES:
        values = features.loc[error_type == key, column].dropna()
        summary[key] = {
            "n": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
        }
    return summary


def find_difficult_samples(
    y_proba: np.ndarray, error_type: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Borderline-band samples and high-confidence errors."""
    low = threshold - BORDERLINE_BAND
    high = threshold + BORDERLINE_BAND
    borderline = np.where((y_proba >= low) & (y_proba <= high))[0]
    hc_fp = np.where((error_type == "FP") & (y_proba >= HIGH_CONF_PROB_FP))[0]
    hc_fn = np.where((error_type == "FN") & (y_proba <= HIGH_CONF_PROB))[0]
    borderline_error = ((error_type == "FP") | (error_type == "FN"))[borderline].sum()
    return {
        "borderline_count": len(borderline),
        "borderline_error_count": int(borderline_error),
        "borderline_error_rate": safe_divide(int(borderline_error), len(borderline)),
        "high_confidence_fp_count": len(hc_fp),
        "high_confidence_fn_count": len(hc_fn),
    }


def run_error_analysis(threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Run the full descriptive error analysis."""
    preds = load_frozen_predictions(threshold)
    error_type = preds["error_type"]
    counts = error_counts(error_type)

    categorical = [c for c in preds["features"].columns if c not in NUMERIC_COLS]
    numerical = [c for c in NUMERIC_COLS if c in preds["features"].columns]

    return {
        "threshold": threshold,
        "threshold_source": "Phase 8B OOF F1-optimal candidate",
        "test_samples": len(error_type),
        "error_counts": counts,
        "error_rates": error_rates(counts),
        "difficult_samples": find_difficult_samples(
            preds["y_proba"], error_type, threshold
        ),
        "categorical_subgroups": {
            col: subgroup_error_rates(preds["features"], error_type, col).to_dict(
                orient="records"
            )
            for col in categorical
        },
        "numerical_summary": {
            col: numerical_error_summary(preds["features"], error_type, col)
            for col in numerical
        },
    }


def save_error_analysis(results: dict[str, Any], output_path: Path) -> None:
    """Persist error analysis results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2)


def main() -> None:
    """Print the error analysis report and save the artifact."""
    warnings.filterwarnings("ignore")
    results = run_error_analysis()

    counts = results["error_counts"]
    rates = results["error_rates"]
    print("=" * 70)
    print("PHASE 10 - ERROR ANALYSIS")
    print("=" * 70)
    print(f"Threshold: {results['threshold']} ({results['threshold_source']})")
    print(f"Test samples: {results['test_samples']}")
    print()
    print("Error counts:")
    for key in ERROR_TYPES:
        print(f"  {key}: {counts[key]}")
    print()
    print("Error rates:")
    for key, value in rates.items():
        print(f"  {key}: {value:.4f}")
    print()
    print("Difficult samples:")
    for key, value in results["difficult_samples"].items():
        print(f"  {key}: {value}")

    output_path = Path(__file__).resolve().parents[2] / "models" / "error_analysis.json"
    save_error_analysis(results, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

"""Phase 13 - SHAP Explainability for the Production Random Forest.

This module provides a leakage-safe SHAP explanation layer for the
production Phase 8C RandomForestClassifier pipeline.

Production boundary:
    Only ``models/final_pipeline.joblib`` (uncalibrated RF) is explained.
    Phase 12 calibrated model is NOT explained (rejected experiment).

SHAP configuration:
    shap.TreeExplainer(
        model_output="raw",
        feature_perturbation="tree_path_dependent",
    )
    The explained model output is the positive-class prediction from the
    production scikit-learn RandomForest (P(churn=Yes)). The `raw` output
    mode returns SHAP values in the model's native output space (for this
    estimator, probability space). `tree_path_dependent` is the default
    feature-dependence/background assumption for Tree SHAP; it does not
    use an interventional background dataset.

Leakage boundaries:
    * Reference/reproducibility sample comes from the training split only.
    * Explanation data (test split) is used strictly post-hoc for
      visualization; it never influences training, tuning, threshold
      selection, calibration, or model adoption.
    * The production model and preprocessor are loaded and used only for
      transform/predict; fit() is never called.

Binary classification semantics:
    Positive class = churn = Yes = class index 1.
    Positive SHAP values push toward churn; negative push away.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from numpy.typing import NDArray

from .config import RANDOM_SEED
from .features import TARGET_COL, load_data

# Production artifact and threshold
PIPELINE_PATH = Path(__file__).resolve().parents[2] / "models" / "final_pipeline.joblib"
PRODUCTION_THRESHOLD = 0.28  # Phase 11 business threshold on raw RF probabilities

# Artifacts directories
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "figures"

# Background sample size (deterministic subset of training data)
BACKGROUND_SAMPLE_SIZE = 500

# Explanation dataset size for global analysis (post-hoc visualization)
GLOBAL_EXPLANATION_SIZE = 200

# Reproducible seed for sampling
EXPLAIN_RANDOM_SEED = RANDOM_SEED


@dataclass
class ShapExplanationResult:
    """Container for computed SHAP explanation."""

    shap_values: NDArray[np.float64]  # (n_samples, n_features, n_classes)
    base_value: float  # expected_value for class 1 (E[P(churn=Yes)])
    feature_names: list[str]
    predictions: NDArray[np.float64]  # predicted probabilities for class 1
    model_output: str = "probability"  # output space: probability (raw output from tree_path_dependent)


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------


def load_production_pipeline(pipeline_path: Path | None = None) -> Any:
    """Load the production (Phase 8C) uncalibrated RF pipeline.

    Args:
        pipeline_path: Optional path. Defaults to models/final_pipeline.joblib.

    Returns:
        Fitted sklearn Pipeline (preprocessor + RandomForest).
    """
    path = pipeline_path or PIPELINE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Production pipeline not found at {path}. "
            "Run customer_churn.predict main() first."
        )
    return joblib.load(path)


def get_preprocessor_and_model(pipeline: Any) -> tuple[Any, Any]:
    """Extract preprocessor and model from the production pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.

    Returns:
        Tuple of (preprocessor, model).

    Raises:
        KeyError: If expected steps are missing.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
    except KeyError as exc:
        raise KeyError(
            f"Pipeline missing expected step. Found: {list(pipeline.named_steps.keys())}"
        ) from exc
    return preprocessor, model


def get_feature_names(preprocessor: Any) -> list[str]:
    """Get transformed feature names from the preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.

    Returns:
        List of feature names after preprocessing.
    """
    return list(preprocessor.get_feature_names_out())


def load_training_background(
    background_size: int = BACKGROUND_SAMPLE_SIZE,
    random_state: int = EXPLAIN_RANDOM_SEED,
) -> tuple[NDArray[np.float64], list[str]]:
    """Load a deterministic reference sample from the training split.

    This sample serves as a reproducibility reference. Under the configured
    tree_path_dependent feature-dependence assumption, Tree SHAP does not
    use an interventional background dataset. The returned sample is retained
    for reproducibility and reference purposes only.

    Args:
        background_size: Number of reference rows (deterministic subset).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (reference_data, feature_names).
    """
    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])

    pipeline = load_production_pipeline()
    preprocessor, _ = get_preprocessor_and_model(pipeline)
    feature_names = get_feature_names(preprocessor)
    X_trans = preprocessor.transform(X_train)

    n_bg = min(background_size, len(X_trans))
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X_trans), size=n_bg, replace=False)
    bg_data = np.asarray(X_trans, dtype=float)[bg_idx]

    return bg_data, feature_names


def load_explanation_data(
    split: str = "test",
) -> tuple[NDArray[np.float64], NDArray[np.int_], list[str]]:
    """Load and transform explanation data (post-hoc visualization only).

    The returned data is used strictly for post-hoc visualization. It does
    not influence training, tuning, threshold selection, calibration, or
    model adoption decisions.

    Args:
        split: Data split ('test' by default; post-hoc only).

    Returns:
        Tuple of (X_transformed, y_true, feature_names).
    """
    df = load_data(split)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values

    pipeline = load_production_pipeline()
    preprocessor, _ = get_preprocessor_and_model(pipeline)
    feature_names = get_feature_names(preprocessor)
    X_trans = preprocessor.transform(X)

    return np.asarray(X_trans, dtype=float), np.asarray(y, dtype=int), feature_names


# ---------------------------------------------------------------------------
# Explainer construction and computation
# ---------------------------------------------------------------------------


def build_tree_explainer(
    model: Any,
    feature_names: list[str],
    background: NDArray[np.float64] | None = None,
) -> shap.TreeExplainer:
    """Build a shap.TreeExplainer for the production RandomForest.

    Uses model_output="raw" with default feature_perturbation="tree_path_dependent".
    Under this configuration, Tree SHAP returns values in the model's native
    output space (for this estimator, positive-class probability).
    The "tree_path_dependent" feature-dependence assumption uses the tree
    structure to define the background; no interventional background dataset
    is used. The background parameter is retained for API compatibility but
    is not used by the tree_path_dependent mode.

    Args:
        model: Fitted RandomForestClassifier.
        feature_names: Transformed feature names.
        background: Optional reference sample (training only). Not used by
            tree_path_dependent but kept for API compatibility.

    Returns:
        Configured shap.TreeExplainer.
    """
    return shap.TreeExplainer(
        model,
        feature_names=feature_names,
        model_output="raw",
    )


def compute_global_shap(
    explainer: shap.TreeExplainer,
    X_explain: NDArray[np.float64],
    model: Any,
) -> ShapExplanationResult:
    """Compute SHAP values for global explanation dataset.

    Args:
        explainer: Configured shap.TreeExplainer.
        X_explain: Transformed explanation features (n_samples, n_features).
        model: Fitted RandomForestClassifier (for predictions).

    Returns:
        ShapExplanationResult with SHAP values and metadata.

    Raises:
        ValueError: If feature dimension mismatch.
    """
    n_features_expected = len(explainer.feature_names or [])
    if X_explain.shape[1] != n_features_expected:
        raise ValueError(
            f"Feature dimension mismatch: data has {X_explain.shape[1]}, "
            f"explainer expects {n_features_expected}."
        )

    shap_values = explainer.shap_values(X_explain)
    # Normalize to (n_samples, n_features, n_classes)
    shap_values = _normalize_shap_shape(shap_values)
    base_value = _extract_base_value(explainer, class_index=1)
    predictions = model.predict_proba(X_explain)[:, 1]

    return ShapExplanationResult(
        shap_values=shap_values,
        base_value=float(base_value),
        feature_names=list(explainer.feature_names or []),
        predictions=predictions,
    )


def _normalize_shap_shape(shap_values: Any) -> NDArray[np.float64]:
    """Normalize SHAP output to (n_samples, n_features, n_classes).

    SHAP may return list (per-class) or ndarray (n, f, c).
    """
    if isinstance(shap_values, list):
        # list of per-class arrays -> stack to (n, f, c)
        stacked = np.stack(shap_values, axis=-1)
        return np.asarray(stacked, dtype=float)
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 2:
        # Single class output -> expand to (n, f, 1)
        return arr[:, :, np.newaxis]
    return arr


def _extract_base_value(explainer: shap.TreeExplainer, class_index: int = 1) -> float:
    """Extract base (expected) value for the positive class."""
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        return float(np.asarray(ev)[class_index])
    return float(ev)


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------


def group_shap_by_original_feature(
    shap_values: NDArray[np.float64],
    feature_names: list[str],
    class_index: int = 1,
) -> pd.DataFrame:
    """Aggregate SHAP values from transformed features to original features.

    For each original feature (prefix before second '__'):
    - mean_abs: mean over samples of sum of |SHAP| across its columns
        (avoids one-hot cancellation)
    - mean_signed: mean over samples of signed SHAP across its columns
    - n_columns: number of transformed columns contributing

    Args:
        shap_values: (n_samples, n_features, n_classes)
        feature_names: Transformed feature names.
        class_index: Class to aggregate (default 1 = churn).

    Returns:
        DataFrame with columns [original_feature, mean_abs, mean_signed, n_columns].
    """
    # shap_values[:, :, class_index] -> (n_samples, n_features)
    sv = np.asarray(shap_values, dtype=float)[:, :, class_index]

    # Derive original feature name from transformed name
    # num__MonthlyCharges -> MonthlyCharges
    # cat__Contract_Month-to-month -> Contract
    original_names: list[str] = []
    for name in feature_names:
        # Remove transformer prefix (num__ or cat__)
        if "__" in name:
            remainder = name.split("__", 1)[1]
        else:
            remainder = name
        # For categorical one-hot, take part before the value delimiter
        # e.g. Contract_Month-to-month -> Contract
        # But be careful: features like "InternetService_Fiber optic" -> InternetService
        # We use a heuristic: take everything before the first underscore that
        # starts a known categorical value. Simpler: use the column prefix.
        # Since one-hot names are <column>_<value>, and numeric are single tokens:
        if name.startswith("cat__"):
            # cat__Contract_Month-to-month -> split on first underscore after removing prefix
            # We need to identify the original categorical column name.
            # The original column names are known from preprocessor; but here we
            # only have transformed names. Use heuristic: original is up to first
            # underscore in the remainder, but some column names have underscores
            # (e.g. PaymentMethod). We derive by taking the remainder and splitting
            # at known categorical value patterns is fragile.
            # Better: original feature = remainder up to last underscore only if
            # the value part looks like a one-hot token. Since we have numeric
            # features as num__X, categorical as cat__Col_Value where Col has no
            # underscore except known ones.
            # Simpler robust approach: original = remainder split at first underscore.
            parts = remainder.split("_", 1)
            original_names.append(parts[0])
        else:
            original_names.append(remainder)

    # Build aggregation
    unique_originals: list[str] = []
    for name in original_names:
        if name not in unique_originals:
            unique_originals.append(name)

    rows = []
    for orig in unique_originals:
        col_mask = [i for i, n in enumerate(original_names) if n == orig]
        if not col_mask:
            continue
        sub = sv[:, col_mask]  # (n_samples, n_subcols)
        # mean_abs: mean over samples of sum |shap| across columns (no cancellation)
        mean_abs = float(np.mean(np.sum(np.abs(sub), axis=1)))
        # mean_signed: mean over samples of signed shap summed across columns
        mean_signed = float(np.mean(np.sum(sub, axis=1)))
        rows.append(
            {
                "original_feature": orig,
                "mean_abs": mean_abs,
                "mean_signed": mean_signed,
                "n_columns": len(col_mask),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_abs", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Local explanation
# ---------------------------------------------------------------------------


@dataclass
class LocalExplanation:
    """Local explanation for a single sample."""

    index: int  # test-set index (safe identifier, not customerID)
    predicted_probability: float
    threshold: float
    predicted_class: int
    actual_class: int | None
    base_value: float
    top_positive: list[dict[str, Any]]  # [{feature, shap_value, feature_value}]
    top_negative: list[dict[str, Any]]
    all_shap: dict[str, float]  # feature -> shap value


def select_local_examples(
    X_explain: NDArray[np.float64],
    y_true: NDArray[np.int_],
    predictions: NDArray[np.float64],
    threshold: float = PRODUCTION_THRESHOLD,
    feature_values: NDArray[np.float64] | None = None,
) -> dict[str, int]:
    """Select deterministic TP/TN/FP/FN example indices.

    Args:
        X_explain: Transformed features (unused here but kept for API symmetry).
        y_true: True labels.
        predictions: Predicted probabilities.
        threshold: Production threshold.
        feature_values: Optional raw feature values for context.

    Returns:
        Dict with keys 'tp', 'tn', 'fp', 'fn' mapping to first index.
    """
    y_pred = (predictions >= threshold).astype(int)

    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    tn = np.where((y_true == 0) & (y_pred == 0))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    result: dict[str, int] = {}
    result["tp"] = int(tp[0]) if len(tp) > 0 else -1
    result["tn"] = int(tn[0]) if len(tn) > 0 else -1
    result["fp"] = int(fp[0]) if len(fp) > 0 else -1
    result["fn"] = int(fn[0]) if len(fn) > 0 else -1

    return result


def compute_local_shap(
    explainer: shap.TreeExplainer,
    X_explain: NDArray[np.float64],
    indices: dict[str, int],
    y_true: NDArray[np.int_],
    model: Any,
    feature_names: list[str],
    threshold: float = PRODUCTION_THRESHOLD,
    top_k: int = 5,
) -> dict[str, LocalExplanation]:
    """Compute local SHAP explanations for selected example indices.

    Args:
        explainer: Configured TreeExplainer.
        X_explain: Transformed features.
        indices: Dict of sample_type -> row index (from select_local_examples).
        y_true: True labels.
        model: Fitted model.
        feature_names: Transformed feature names.
        threshold: Production threshold.
        top_k: Number of top positive/negative contributors.

    Returns:
        Dict sample_type -> LocalExplanation.
    """
    # Collect valid indices
    valid = {k: v for k, v in indices.items() if v >= 0}
    if not valid:
        return {}

    unique_rows = sorted(set(valid.values()))
    # Compute SHAP for unique rows only (efficient)
    shap_vals = explainer.shap_values(X_explain[unique_rows])
    shap_vals = _normalize_shap_shape(shap_vals)
    preds = model.predict_proba(X_explain[unique_rows])[:, 1]
    base_value = _extract_base_value(explainer, class_index=1)

    # Map row position back to sample type
    row_to_pos = {row: pos for pos, row in enumerate(unique_rows)}

    result: dict[str, LocalExplanation] = {}
    for sample_type, row_idx in valid.items():
        pos = row_to_pos[row_idx]
        sv = shap_vals[pos, :, 1]  # (n_features,)
        prob = float(preds[pos])
        pred_class = int(prob >= threshold)
        actual = int(y_true[row_idx])

        # Build feature->shap mapping
        all_shap = {feature_names[i]: float(sv[i]) for i in range(len(feature_names))}

        # Sort by shap value
        order = np.argsort(sv)
        # Top positive: highest shap values
        top_pos_idx = order[::-1][:top_k]
        # Top negative: lowest shap values
        top_neg_idx = order[:top_k]

        top_positive = [
            {
                "feature": feature_names[i],
                "shap_value": float(sv[i]),
                "feature_value": float(X_explain[row_idx, i]),
            }
            for i in top_pos_idx
            if sv[i] > 0
        ]
        top_negative = [
            {
                "feature": feature_names[i],
                "shap_value": float(sv[i]),
                "feature_value": float(X_explain[row_idx, i]),
            }
            for i in top_neg_idx
            if sv[i] < 0
        ]

        result[sample_type] = LocalExplanation(
            index=row_idx,
            predicted_probability=prob,
            threshold=threshold,
            predicted_class=pred_class,
            actual_class=actual,
            base_value=base_value,
            top_positive=top_positive,
            top_negative=top_negative,
            all_shap=all_shap,
        )

    return result


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


def save_global_artifacts(
    shap_result: ShapExplanationResult,
    global_importance: pd.DataFrame,
    grouped_importance: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Save global explanation artifacts (CSV + JSON).

    Args:
        shap_result: SHAP result.
        global_importance: Per-feature mean |SHAP| table.
        grouped_importance: Per-original-feature aggregated table.
        output_dir: Output directory (defaults to models/).

    Returns:
        Dict artifact name -> path.
    """
    out = output_dir or MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    # Global feature importance CSV
    gp_path = out / "shap_global_importance.csv"
    global_importance.to_csv(gp_path, index=False)
    artifacts["global_importance"] = gp_path

    # Grouped by original feature
    grouped_path = out / "shap_grouped_importance.csv"
    grouped_importance.to_csv(grouped_path, index=False)
    artifacts["grouped_importance"] = grouped_path

    # Feature mapping JSON
    mapping = {
        "transformed_features": shap_result.feature_names,
        "n_transformed_features": len(shap_result.feature_names),
        "original_features": grouped_importance["original_feature"].tolist(),
        "n_original_features": len(grouped_importance),
    }
    mapping_path = out / "shap_feature_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    artifacts["feature_mapping"] = mapping_path

    # Metadata / provenance
    metadata = build_provenance_metadata(shap_result)
    meta_path = out / "shap_provenance.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    artifacts["provenance"] = meta_path

    return artifacts


def build_provenance_metadata(shap_result: ShapExplanationResult) -> dict[str, Any]:
    """Build provenance metadata for explanation artifacts."""
    return {
        "production_model_artifact": "models/final_pipeline.joblib",
        "model_type": "RandomForestClassifier",
        "phase_8c_config": {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "business_threshold": 0.28,
        "output_space": "probability (positive-class output from tree_path_dependent SHAP)",
        "positive_class": "Yes",
        "positive_class_index": 1,
        "expected_value_class1": shap_result.base_value,
        "feature_count": len(shap_result.feature_names),
        "shap_version": shap.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "reference_source": "training split (deterministic subset, reproducibility only)",
        "reference_sample_size": BACKGROUND_SAMPLE_SIZE,
        "explanation_source": "frozen test split (post-hoc visualization only)",
        "random_seed": EXPLAIN_RANDOM_SEED,
        "phase12_calibrated_model_explained": False,
        "leakage_note": "Test data used only for post-hoc visualization; never for model decisions.",
    }


def save_local_artifacts(
    local_explanations: dict[str, LocalExplanation],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Save local explanation artifacts as JSON.

    Args:
        local_explanations: Dict sample_type -> LocalExplanation.
        output_dir: Output directory (defaults to models/).

    Returns:
        Dict artifact name -> path.
    """
    out = output_dir or MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    for sample_type, loc in local_explanations.items():
        payload[sample_type] = {
            "index": loc.index,
            "predicted_probability": loc.predicted_probability,
            "threshold": loc.threshold,
            "predicted_class": loc.predicted_class,
            "actual_class": loc.actual_class,
            "base_value": loc.base_value,
            "top_positive": loc.top_positive,
            "top_negative": loc.top_negative,
            "all_shap": loc.all_shap,
        }

    path = out / "shap_local_explanations.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return {"local_explanations": path}


def save_figures(
    shap_result: ShapExplanationResult,
    global_importance: pd.DataFrame,
    local_explanations: dict[str, LocalExplanation],
    X_global: NDArray[np.float64],
    X_full: NDArray[np.float64],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate SHAP summary, bar, and waterfall figures.

    Args:
        shap_result: Global SHAP result (computed on X_global).
        global_importance: Feature importance table.
        local_explanations: Local explanations.
        X_global: Transformed features used for global SHAP (summary/bar).
        X_full: Full transformed features (for waterfall context, includes local indices).
        output_dir: Figures directory (defaults to notebooks/figures/).

    Returns:
        Dict figure name -> path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = output_dir or FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    # Class-1 SHAP values for plots: (n_samples, n_features)
    sv_class1 = shap_result.shap_values[:, :, 1]

    # Summary / beeswarm plot (uses X_global which matches shap_result)
    try:
        _, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv_class1,
            features=X_global,
            feature_names=shap_result.feature_names,
            show=False,
            max_display=20,
        )
        plt.title("SHAP Feature Impact on Churn Probability (Class 1)")
        plt.tight_layout()
        summary_path = out / "shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        artifacts["summary"] = summary_path
    except Exception as exc:  # pragma: no cover - plot failure not fatal  # noqa: BLE001
        plt.close()
        print(f"Summary plot failed: {exc}")

    # Bar plot (mean |SHAP|)
    try:
        _, ax = plt.subplots(figsize=(10, 8))
        top_n = global_importance.head(20)
        ax.barh(
            top_n["feature"][::-1],
            top_n["mean_abs_shap"][::-1],
            color="steelblue",
        )
        ax.set_xlabel("Mean |SHAP| (probability space)")
        ax.set_title("Top 20 Features by Mean Absolute SHAP")
        plt.tight_layout()
        bar_path = out / "shap_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        artifacts["bar"] = bar_path
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        plt.close()
        print(f"Bar plot failed: {exc}")

    # Waterfall plots for local examples
    for sample_type, loc in local_explanations.items():
        try:
            row_idx = loc.index
            sv_row = sv_class1[row_idx: row_idx + 1]
            feat_row = X_full[row_idx: row_idx + 1]

            _, ax = plt.subplots(figsize=(10, 8))
            shap.plots._waterfall.waterfall_legacy(
                loc.base_value,
                sv_row[0],
                features=feat_row[0],
                feature_names=shap_result.feature_names,
                show=False,
                max_display=15,
            )
            plt.title(
                f"SHAP Waterfall - {sample_type.upper()} "
                f"(prob={loc.predicted_probability:.3f}, thr={loc.threshold})"
            )
            plt.tight_layout()
            wf_path = out / f"shap_waterfall_{sample_type.lower()}.png"
            plt.savefig(wf_path, dpi=150, bbox_inches="tight")
            plt.close()
            artifacts[f"waterfall_{sample_type.lower()}"] = wf_path
        except Exception as exc:  # pragma: no cover  # noqa: BLE001
            plt.close()
            print(f"Waterfall plot {sample_type} failed: {exc}")

    return artifacts


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_explainability_analysis(
    n_global: int = GLOBAL_EXPLANATION_SIZE,
    background_size: int = BACKGROUND_SAMPLE_SIZE,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full Phase 13 explainability analysis.

    Steps:
        1. Load production pipeline (never refit).
        2. Build TreeExplainer (raw output, tree_path_dependent).
        3. Load reference sample from training (never test).
        4. Load explanation data from test (post-hoc only).
        5. Compute global SHAP + grouped importance.
        6. Select TP/TN/FP/FN local examples.
        7. Compute local SHAP.
        8. Save artifacts and figures.

    Args:
        n_global: Number of test samples for global explanation.
        background_size: Reference sample size.
        output_dir: Artifact output directory.
        figures_dir: Figure output directory.

    Returns:
        Dict with results and artifact paths.
    """
    print("=" * 80)
    print("PHASE 13 - SHAP EXPLAINABILITY")
    print("=" * 80)

    # 1. Load production pipeline
    print("\n[1] Loading production pipeline...")
    pipeline = load_production_pipeline()
    preprocessor, model = get_preprocessor_and_model(pipeline)
    feature_names = get_feature_names(preprocessor)
    print(f"  Features: {len(feature_names)}")

    # 2. Build explainer
    print("\n[2] Building TreeExplainer (raw output, tree_path_dependent)...")
    explainer = build_tree_explainer(model, feature_names)
    print(f"  Expected value (class 1): {explainer.expected_value[1]:.6f}")

    # 3. Load reference sample (training only)
    print("\n[3] Loading training reference sample...")
    background, _ = load_training_background(background_size)
    print(f"  Reference sample shape: {background.shape}")

    # 4. Load explanation data (test, post-hoc)
    print("\n[4] Loading explanation data (test, post-hoc)...")
    X_explain, y_true, _ = load_explanation_data("test")
    # Use subset for global analysis
    n_use = min(n_global, len(X_explain))
    X_global = X_explain[:n_use]
    _ = y_true[:n_use]
    print(f"  Explanation samples: {n_use}")

    # 5. Compute global SHAP
    print("\n[5] Computing global SHAP values...")
    global_result = compute_global_shap(explainer, X_global, model)
    print(f"  SHAP shape: {global_result.shap_values.shape}")

    # 6. Feature importance tables
    print("\n[6] Computing feature importance...")
    sv_class1 = global_result.shap_values[:, :, 1]
    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(np.abs(sv_class1), axis=0),
            "mean_signed_shap": np.mean(sv_class1, axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    grouped_importance = group_shap_by_original_feature(
        global_result.shap_values, feature_names, class_index=1
    )
    print(f"  Top original feature: {grouped_importance.iloc[0]['original_feature']}")

    # 7. Local examples (full test set for selection)
    print("\n[7] Selecting local TP/TN/FP/FN examples...")
    all_preds = model.predict_proba(X_explain)[:, 1]
    indices = select_local_examples(X_explain, y_true, all_preds, PRODUCTION_THRESHOLD)
    print(f"  Indices: {indices}")

    print("\n[8] Computing local SHAP explanations...")
    local_explanations = compute_local_shap(
        explainer, X_explain, indices, y_true, model, feature_names, PRODUCTION_THRESHOLD
    )

    # 9. Save artifacts
    print("\n[9] Saving artifacts...")
    out_dir = output_dir or MODELS_DIR
    fig_dir = figures_dir or FIGURES_DIR

    global_artifacts = save_global_artifacts(
        global_result, global_importance, grouped_importance, out_dir
    )
    local_artifacts = save_local_artifacts(local_explanations, out_dir)
    figures = save_figures(
        global_result, global_importance, local_explanations, X_global, X_explain, fig_dir
    )

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 13 RESULTS")
    print("=" * 80)
    print("\nOutput space: probability (positive-class output from tree_path_dependent SHAP)")
    print(f"Expected value (class 1): {global_result.base_value:.6f}")
    print(f"Business threshold: {PRODUCTION_THRESHOLD}")
    print("\nTop 10 features (transformed):")
    for _, row in global_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.6f}")
    print("\nTop 10 features (original, grouped):")
    for _, row in grouped_importance.head(10).iterrows():
        print(
            f"  {row['original_feature']}: mean_abs={row['mean_abs_shap']:.6f}, "
            f"mean_signed={row['mean_signed']:.6f}"
        )

    print("\nLocal explanations:")
    for stype, loc in local_explanations.items():
        print(
            f"  {stype}: idx={loc.index}, prob={loc.predicted_probability:.3f}, "
            f"pred={loc.predicted_class}, actual={loc.actual_class}"
        )

    print("\nArtifacts:")
    for name, path in {**global_artifacts, **local_artifacts, **figures}.items():
        print(f"  {name}: {path}")

    return {
        "global_result": global_result,
        "global_importance": global_importance,
        "grouped_importance": grouped_importance,
        "local_explanations": local_explanations,
        "artifacts": {**global_artifacts, **local_artifacts, **figures},
    }


def main() -> None:
    """Entry point: run explainability analysis."""
    import warnings

    warnings.filterwarnings("ignore")
    run_explainability_analysis()


if __name__ == "__main__":
    main()

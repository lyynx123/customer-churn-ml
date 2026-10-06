"""Phase 12 - Probability Calibration.

Post-hoc probability calibration for the frozen Phase 8C RandomForest
classifier. The calibrator maps a raw RF probability scalar to a
calibrated probability scalar.

Architecture (leakage-safe):
    Frozen Phase 8C RF pipeline (fitted on full train)
        -> cross_val_predict -> OOF raw probabilities
        -> cross-fitted probability calibrator
        -> calibrated probability

Method selection is based on cross-fitted calibration metrics only:
    primary: cross-fitted Log Loss
    secondary: cross-fitted Brier Score
    tie-breaker: prefer sigmoid (simpler, 2 parameters)

Business cost optimization is NEVER used for method selection.
Frozen test data is NEVER used for calibration fitting or method
selection.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .config import RANDOM_SEED
from .features import TARGET_COL, load_data
from .final_evaluation import build_final_pipeline

# Tolerance for practically-equivalent calibration metrics
LOG_LOSS_TOLERANCE: float = 1e-4
BRIER_TOLERANCE: float = 1e-4

CV_FOLDS: int = 5

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CALIBRATION_ANALYSIS_PATH = MODELS_DIR / "calibration_analysis.json"


class SigmoidProbabilityCalibrator:
    """LogisticRegression calibrator on logit-transformed raw probabilities.

    Maps raw RF probability -> calibrated probability via:
        logit(p_raw) -> LogisticRegression -> sigmoid output

    Handles p=0 and p=1 safely with clipping before logit.
    """

    def __init__(self, clip_eps: float = 1e-6):
        self.clip_eps = float(clip_eps)
        self.calibrator = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
        self._fitted = False

    @staticmethod
    def _logit(p: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
        """Transform probabilities to log-odds, clipping to (eps, 1-eps)."""
        p_clipped = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return np.log(p_clipped / (1.0 - p_clipped))

    def fit(
        self, raw_probability: NDArray[np.float64], y_true: NDArray[np.int_]
    ) -> SigmoidProbabilityCalibrator:
        """Fit the calibrator on raw probabilities and true labels.

        Args:
            raw_probability: Raw model probabilities in [0, 1].
            y_true: True binary labels (0/1).

        Returns:
            Self.
        """
        if len(raw_probability) != len(y_true):
            raise ValueError(
                f"Length mismatch: raw_probability={len(raw_probability)}, "
                f"y_true={len(y_true)}. Both arrays must have equal length."
            )
        if len(raw_probability) == 0:
            raise ValueError("Cannot fit calibrator on empty arrays.")
        if not np.all((raw_probability >= 0) & (raw_probability <= 1)):
            raise ValueError(
                "Raw probabilities must be in [0, 1]. Found values outside range."
            )
        if not set(np.unique(y_true)).issubset({0, 1}):
            raise ValueError(
                f"y_true contains labels other than 0/1: {np.unique(y_true)}"
            )

        logits = self._logit(raw_probability, self.clip_eps).reshape(-1, 1)
        self.calibrator.fit(logits, np.asarray(y_true))
        self._fitted = True
        return self

    def predict_proba(
        self, raw_probability: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return calibrated probabilities for the positive class.

        Args:
            raw_probability: Raw model probabilities in [0, 1].

        Returns:
            Calibrated positive-class probabilities in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted yet. Call fit() first.")
        if not np.all((raw_probability >= 0) & (raw_probability <= 1)):
            raise ValueError(
                "Raw probabilities must be in [0, 1]. Found values outside range."
            )
        logits = self._logit(raw_probability, self.clip_eps).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]

    def __getstate__(self) -> dict[str, Any]:
        """Ensure calibrator can be serialized via joblib."""
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state after joblib deserialization."""
        self.__dict__.update(state)


class IsotonicProbabilityCalibrator:
    """IsotonicRegression calibrator on raw probabilities.

    Maps raw RF probability -> calibrated probability using an
    isotonic (monotonic) fit. Bounds clamped to [0, 1] with
    out-of-bounds clipping.
    """

    def __init__(self):
        self.regressor = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
            increasing=True,
        )
        self._fitted = False

    def fit(
        self, raw_probability: NDArray[np.float64], y_true: NDArray[np.int_]
    ) -> IsotonicProbabilityCalibrator:
        """Fit the calibrator on raw probabilities and true labels.

        Args:
            raw_probability: Raw model probabilities in [0, 1].
            y_true: True binary labels (0/1).

        Returns:
            Self.
        """
        if len(raw_probability) != len(y_true):
            raise ValueError(
                f"Length mismatch: raw_probability={len(raw_probability)}, "
                f"y_true={len(y_true)}. Both arrays must have equal length."
            )
        if len(raw_probability) == 0:
            raise ValueError("Cannot fit calibrator on empty arrays.")
        if not np.all((raw_probability >= 0) & (raw_probability <= 1)):
            raise ValueError(
                "Raw probabilities must be in [0, 1]. Found values outside range."
            )
        if not set(np.unique(y_true)).issubset({0, 1}):
            raise ValueError(
                f"y_true contains labels other than 0/1: {np.unique(y_true)}"
            )

        self.regressor.fit(
            np.asarray(raw_probability, dtype=float),
            np.asarray(y_true, dtype=int),
        )
        self._fitted = True
        return self

    def predict_proba(
        self, raw_probability: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return calibrated probabilities for the positive class.

        Args:
            raw_probability: Raw model probabilities in [0, 1].

        Returns:
            Calibrated positive-class probabilities in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted yet. Call fit() first.")
        if not np.all((raw_probability >= 0) & (raw_probability <= 1)):
            raise ValueError(
                "Raw probabilities must be in [0, 1]. Found values outside range."
            )
        return np.asarray(self.regressor.predict(raw_probability), dtype=float)

    def __getstate__(self) -> dict[str, Any]:
        """Ensure calibrator can be serialized via joblib."""
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state after joblib deserialization."""
        self.__dict__.update(state)


def build_sigmoid_calibrator() -> SigmoidProbabilityCalibrator:
    """Construct a sigmoid (Platt) probability calibrator."""
    return SigmoidProbabilityCalibrator()


def build_isotonic_calibrator() -> IsotonicProbabilityCalibrator:
    """Construct an isotonic probability calibrator."""
    return IsotonicProbabilityCalibrator()


def load_train_xy() -> tuple[pd.DataFrame, NDArray[np.int_]]:
    """Load frozen training features and binary labels (train split only)."""

    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = np.asarray(
        train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values
    )
    return X_train, y_train


def get_oof_raw_probabilities(
    n_splits: int = CV_FOLDS,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Generate genuine out-of-fold raw probabilities.

    Constructs an UNFITTED Phase 8C-spec pipeline and runs
    cross_val_predict on the training partition only. The final
    full-training RF is NEVER fitted before OOF generation.

    Args:
        n_splits: Number of StratifiedKFold splits.

    Returns:
        Tuple of (y_true, y_proba_raw_oof).
    """
    X_train, y_train = load_train_xy()

    pipeline = build_final_pipeline()  # UNFITTED Phase 8C-spec
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )

    from sklearn.model_selection import cross_val_predict

    y_proba_oof = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return y_train, np.asarray(y_proba_oof, dtype=float)


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a single method across cross-fitted folds."""

    method: str
    log_loss_folds: list[float] = field(default_factory=list)
    brier_folds: list[float] = field(default_factory=list)
    roc_auc_folds: list[float] = field(default_factory=list)
    pr_auc_folds: list[float] = field(default_factory=list)
    calibrated_oof_proba: NDArray[np.float64] | None = None

    @property
    def mean_log_loss(self) -> float:
        return float(np.mean(self.log_loss_folds)) if self.log_loss_folds else float("nan")

    @property
    def mean_brier(self) -> float:
        return float(np.mean(self.brier_folds)) if self.brier_folds else float("nan")

    @property
    def mean_roc_auc(self) -> float:
        return float(np.mean(self.roc_auc_folds)) if self.roc_auc_folds else float("nan")

    @property
    def mean_pr_auc(self) -> float:
        return float(np.mean(self.pr_auc_folds)) if self.pr_auc_folds else float("nan")


def _calibrate_cross_fitted(
    y_true: NDArray[np.int_],
    raw_proba_oof: NDArray[np.float64],
    calibrator_builder: Any,
    n_splits: int = CV_FOLDS,
) -> CalibrationMetrics:
    """Cross-fitted calibration evaluation for one calibrator type.

    For each outer fold:
        - calibrator is fitted on the other 4 folds' OOF raw probabilities
        - calibrated probabilities are produced for the held-out fold
        - calibration metrics (log loss, brier) are evaluated on the
          held-out fold

    Discrimination metrics (ROC-AUC, PR-AUC) are computed as sanity
    checks only, never used for method selection.

    Args:
        y_true: True labels for the OOF data (training partition only).
        raw_proba_oof: OOF raw probabilities (training partition only).
        calibrator_builder: Callable returning an unfitted calibrator.
        n_splits: Number of cross-fitting folds.

    Returns:
        CalibrationMetrics with per-fold log loss/brier and calibrated
        OOF probabilities covering the entire training partition.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    calibrated_oof = np.zeros_like(raw_proba_oof, dtype=float)

    metrics = CalibrationMetrics(method=calibrator_builder.__name__)
    for train_idx, val_idx in cv.split(raw_proba_oof, y_true):
        calibrator = calibrator_builder()
        calibrator.fit(raw_proba_oof[train_idx], y_true[train_idx])
        p_cal = calibrator.predict_proba(raw_proba_oof[val_idx])
        calibrated_oof[val_idx] = p_cal

        metrics.log_loss_folds.append(
            float(log_loss(y_true[val_idx], p_cal))
        )
        metrics.brier_folds.append(
            float(brier_score_loss(y_true[val_idx], p_cal))
        )
        metrics.roc_auc_folds.append(
            float(roc_auc_score(y_true[val_idx], p_cal))
        )
        metrics.pr_auc_folds.append(
            float(average_precision_score(y_true[val_idx], p_cal))
        )

    metrics.calibrated_oof_proba = calibrated_oof
    return metrics


def cross_fit_calibration_evaluation(
    y_true: NDArray[np.int_],
    raw_proba_oof: NDArray[np.float64],
    n_splits: int = CV_FOLDS,
) -> tuple[CalibrationMetrics, CalibrationMetrics, CalibrationMetrics]:
    """Evaluate sigmoid and isotonic calibration via cross-fitting.

    Also returns the uncalibrated (baseline) metrics as a third element.

    Args:
        y_true: True labels (training partition only).
        raw_proba_oof: OOF raw probabilities (training partition only).
        n_splits: Number of cross-fitting folds.

    Returns:
        Tuple of (sigmoid_metrics, isotonic_metrics, baseline_metrics).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    baseline = CalibrationMetrics(method="baseline")
    for train_idx, val_idx in cv.split(raw_proba_oof, y_true):
        p_raw = raw_proba_oof[val_idx]
        baseline.log_loss_folds.append(float(log_loss(y_true[val_idx], p_raw)))
        baseline.brier_folds.append(float(brier_score_loss(y_true[val_idx], p_raw)))
        baseline.roc_auc_folds.append(float(roc_auc_score(y_true[val_idx], p_raw)))
        baseline.pr_auc_folds.append(
            float(average_precision_score(y_true[val_idx], p_raw))
        )

    sigmoid_metrics = _calibrate_cross_fitted(
        y_true, raw_proba_oof, build_sigmoid_calibrator, n_splits
    )
    isotonic_metrics = _calibrate_cross_fitted(
        y_true, raw_proba_oof, build_isotonic_calibrator, n_splits
    )

    return sigmoid_metrics, isotonic_metrics, baseline


def select_calibration_method(
    sigmoid_metrics: CalibrationMetrics,
    isotonic_metrics: CalibrationMetrics,
    log_loss_tolerance: float = LOG_LOSS_TOLERANCE,
    brier_tolerance: float = BRIER_TOLERANCE,
) -> tuple[str, dict[str, Any]]:
    """Deterministically select a calibration method.

    Selection protocol (pre-defined, no test data):
        1. Compare mean cross-fitted Log Loss (primary).
           Lower wins.
        2. If Log Loss within tolerance, compare mean cross-fitted
           Brier Score (secondary). Lower wins.
        3. If still practically equivalent, prefer sigmoid (simpler).

    Business cost is NEVER used for method selection.

    Args:
        sigmoid_metrics: Cross-fitted metrics for sigmoid calibrator.
        isotonic_metrics: Cross-fitted metrics for isotonic calibrator.
        log_loss_tolerance: Tolerance for practically-equivalent Log Loss.
        brier_tolerance: Tolerance for practically-equivalent Brier Score.

    Returns:
        Tuple of (selected_method_name, selection_rationale).
    """
    sig_log_loss = sigmoid_metrics.mean_log_loss
    iso_log_loss = isotonic_metrics.mean_log_loss

    rationale: dict[str, Any] = {
        "protocol": (
            "1. min cross-fitted Log Loss (primary); "
            "2. min cross-fitted Brier Score (secondary); "
            "3. prefer sigmoid on tie"
        ),
        "sigmoid_log_loss": sig_log_loss,
        "isotonic_log_loss": iso_log_loss,
        "sigmoid_brier": sigmoid_metrics.mean_brier,
        "isotonic_brier": isotonic_metrics.mean_brier,
        "log_loss_tolerance": log_loss_tolerance,
        "brier_tolerance": brier_tolerance,
    }

    if abs(sig_log_loss - iso_log_loss) > log_loss_tolerance:
        selected = "sigmoid" if sig_log_loss < iso_log_loss else "isotonic"
        rationale["decision_rule_applied"] = "log_loss"
    else:
        sig_brier = sigmoid_metrics.mean_brier
        iso_brier = isotonic_metrics.mean_brier
        if abs(sig_brier - iso_brier) > brier_tolerance:
            selected = "sigmoid" if sig_brier < iso_brier else "isotonic"
            rationale["decision_rule_applied"] = "brier"
        else:
            selected = "sigmoid"
            rationale["decision_rule_applied"] = "tie_break_prefer_sigmoid"

    rationale["selected_method"] = selected
    return selected, rationale


def compute_calibration_metrics_table(
    sigmoid_metrics: CalibrationMetrics,
    isotonic_metrics: CalibrationMetrics,
    baseline_metrics: CalibrationMetrics,
) -> dict[str, Any]:
    """Build a machine-readable calibration metrics comparison table.

    All metrics are computed on cross-fitted training data only.
    The frozen test set is never used for this comparison.
    """
    return {
        "baseline": {
            "log_loss": baseline_metrics.mean_log_loss,
            "brier_score": baseline_metrics.mean_brier,
            "roc_auc": baseline_metrics.mean_roc_auc,
            "pr_auc": baseline_metrics.mean_pr_auc,
            "per_fold_log_loss": baseline_metrics.log_loss_folds,
            "per_fold_brier": baseline_metrics.brier_folds,
        },
        "sigmoid": {
            "log_loss": sigmoid_metrics.mean_log_loss,
            "brier_score": sigmoid_metrics.mean_brier,
            "roc_auc": sigmoid_metrics.mean_roc_auc,
            "pr_auc": sigmoid_metrics.mean_pr_auc,
            "per_fold_log_loss": sigmoid_metrics.log_loss_folds,
            "per_fold_brier": sigmoid_metrics.brier_folds,
        },
        "isotonic": {
            "log_loss": isotonic_metrics.mean_log_loss,
            "brier_score": isotonic_metrics.mean_brier,
            "roc_auc": isotonic_metrics.mean_roc_auc,
            "pr_auc": isotonic_metrics.mean_pr_auc,
            "per_fold_log_loss": isotonic_metrics.log_loss_folds,
            "per_fold_brier": isotonic_metrics.brier_folds,
        },
    }


def build_final_calibrator(
    selected_method: str,
    y_true: NDArray[np.int_],
    raw_proba_oof: NDArray[np.float64],
) -> SigmoidProbabilityCalibrator | IsotonicProbabilityCalibrator:
    """Fit the selected calibrator on ALL OOF raw probabilities + labels.

    Args:
        selected_method: "sigmoid" or "isotonic".
        y_true: True labels for the OOF data.
        raw_proba_oof: OOF raw probabilities.

    Returns:
        Fitted calibrator (SigmoidProbabilityCalibrator or
        IsotonicProbabilityCalibrator).
    """
    if selected_method == "sigmoid":
        calibrator: SigmoidProbabilityCalibrator | IsotonicProbabilityCalibrator = build_sigmoid_calibrator()
    elif selected_method == "isotonic":
        calibrator = build_isotonic_calibrator()
    else:
        raise ValueError(
            f"Unknown calibration method '{selected_method}'. "
            "Expected 'sigmoid' or 'isotonic'."
        )
    return calibrator.fit(raw_proba_oof, y_true)


def run_calibration_analysis() -> dict[str, Any]:
    """Run the full Phase 12 calibration analysis.

    Steps:
        1. Generate OOF raw probabilities (unfitted pipeline, train only).
        2. Cross-fit sigmoid and isotonic calibrators.
        3. Select method via deterministic protocol.
        4. Return comparison table and selection rationale.

    Returns:
        Dictionary with all calibration analysis results.
    """
    warnings.filterwarnings("ignore")

    y_true, raw_proba_oof = get_oof_raw_probabilities()

    sigmoid_metrics, isotonic_metrics, baseline_metrics = (
        cross_fit_calibration_evaluation(y_true, raw_proba_oof)
    )

    selected_method, rationale = select_calibration_method(
        sigmoid_metrics, isotonic_metrics
    )

    comparison = compute_calibration_metrics_table(
        sigmoid_metrics, isotonic_metrics, baseline_metrics
    )

    return {
        "selected_method": selected_method,
        "selection_rationale": rationale,
        "metrics_comparison": comparison,
        "data_provenance": {
            "data_source": "OOF (5-fold StratifiedKFold cross_val_predict on training data)",
            "test_set_used": False,
            "cv_config": {
                "n_splits": CV_FOLDS,
                "shuffle": True,
                "random_state": RANDOM_SEED,
            },
        },
        "frozen_test_used_for_calibration": False,
    }


def save_calibration_analysis(results: dict[str, Any], output_path: Path) -> None:
    """Persist calibration analysis results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2)


def main() -> None:
    """Run calibration analysis and save artifacts."""
    results = run_calibration_analysis()

    print("=" * 80)
    print("PHASE 12 - CALIBRATION ANALYSIS")
    print("=" * 80)
    print(f"Selected method: {results['selected_method']}")
    print(f"Rationale: {results['selection_rationale']}")
    print("\nMetrics comparison (cross-fitted, training data only):")
    for method, metrics in results["metrics_comparison"].items():
        print(f"  {method}: log_loss={metrics['log_loss']:.6f}, "
              f"brier={metrics['brier_score']:.6f}, "
              f"roc_auc={metrics['roc_auc']:.6f}, "
              f"pr_auc={metrics['pr_auc']:.6f}")

    save_calibration_analysis(results, CALIBRATION_ANALYSIS_PATH)
    print(f"\nSaved to {CALIBRATION_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()

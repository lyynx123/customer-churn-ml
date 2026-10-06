"""Custom calibrated RF estimator for inference.

The final production artifact is an instance of CalibratedRFClassifier
that contains:
  1. Preprocessor (fitted on the training partition).
  2. RandomForestClassifier (Phase 8C spec, fitted on the full training set).
  3. Probability calibrator (Sigmoid or Isotonic), fitted on all OOF data.

The estimator exposes sklearn-like predict_proba() that does **not** refit the
calibrator or the RF model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from .features import TARGET_COL, build_preprocessor
from .final_evaluation import FINAL_RF_CONFIG

# Import RandomForestClassifier lazily to avoid heavy import at module load.
RandomForestClassifier = None  # type: ignore

try:
    from sklearn.ensemble import RandomForestClassifier as _RF
    RandomForestClassifier = _RF
except ImportError:  # pragma: no cover
    pass

# Calibrators will be imported in the training code, but we keep the import
# optional to avoid circular dependencies for the runtime.

@dataclass
class CalibratedRFClassifier(BaseEstimator):
    """Wrapper that applies a preprocessor, RF, and a probability calibrator.

    Parameters are expected to be already fitted. The class does *not* implement
    a fit method to preserve calibration integrity. Only predict_proba() is
    implemented and is deterministic.
    """

    preprocessor: Any
    rf: Any
    calibrator: Any

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated probabilities for the positive class.

        The method is strict: it does not refit any component and
        expects ``self.rf`` support predict_proba.
        """
        # Transform features
        X_trans = self.preprocessor.transform(X)
        # Raw RF probabilities
        raw = self.rf.predict_proba(X_trans)[:, 1]
        # Calibrate
        cal = self.calibrator.predict_proba(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction using calibrated probabilities."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def fit(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("CalibratedRFClassifier does not support fitting; use fit on needed components beforehand.")

    def get_params(self, deep=True):  # pragma: no cover
        # Required for sklearn compatibility; we expose subcomponents.
        return {
            "preprocessor": self.preprocessor,
            "rf": self.rf,
            "calibrator": self.calibrator,
        }

    def set_params(self, **params):  # pragma: no cover
        raise RuntimeError("CalibratedRFClassifier does not allow setting params after construction.")

# Utility functions to build/serialise the final artifact

from .calibration import IsotonicProbabilityCalibrator, SigmoidProbabilityCalibrator

CalibratorType = SigmoidProbabilityCalibrator | IsotonicProbabilityCalibrator

# Utility functions to build/serialise the final artifact

def build_calibrated_estimator(
    raw_proba_oof: np.ndarray, y_true: np.ndarray, selected_method: str
) -> CalibratedRFClassifier:
    """Construct a CalibratedRFClassifier after calibration is finalized.

    Parameters:
        raw_proba_oof: OOF raw probabilities used to fit the calibrator.
        y_true: Corresponding true labels.
        selected_method: "sigmoid" or "isotonic".
    """
    # Build preprocessor
    preprocessor = build_preprocessor()
    # Load data
    from .features import load_data

    train_df = load_data("train")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = np.asarray(train_df[TARGET_COL].replace({"No": 0, "Yes": 1}).astype(int).values)
    # Fit RF on full training
    rf = RandomForestClassifier(**FINAL_RF_CONFIG)  # type: ignore[misc, arg-type]
    rf.fit(preprocessor.fit_transform(X_train), y_train)
    # Fit calibrator on all OOF data
    if selected_method == "sigmoid":
        calibrator: CalibratorType = SigmoidProbabilityCalibrator()
    elif selected_method == "isotonic":
        calibrator = IsotonicProbabilityCalibrator()
    else:
        raise ValueError("Unknown calibrator method")
    calibrator.fit(raw_proba_oof, y_true)
    return CalibratedRFClassifier(preprocessor, rf, calibrator)

# The actual training script will import this function and save the result
# with joblib.dump after invoking calibration analysis.

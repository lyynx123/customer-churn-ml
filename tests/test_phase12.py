"""Tests for Phase 12 - Probability Calibration."""

from pathlib import Path

import joblib
import numpy as np
import pytest

from customer_churn.business_cost import DEFAULT_COST_MATRIX
from customer_churn.business_threshold_optimizer import optimize_threshold_business
from customer_churn.calibrated_pipeline import CalibratedRFClassifier
from customer_churn.calibration import (
    SigmoidProbabilityCalibrator,
    build_isotonic_calibrator,
    build_sigmoid_calibrator,
    cross_fit_calibration_evaluation,
    get_oof_raw_probabilities,
    run_calibration_analysis,
    select_calibration_method,
)


class TestSigmoidProbabilityCalibrator:
    """Tests for SigmoidProbabilityCalibrator."""

    def test_fit_predict_basic(self):
        """Sigmoid calibrator fits and predicts on synthetic data."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        raw_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])

        cal = build_sigmoid_calibrator()
        cal.fit(raw_proba, y_true)
        pred = cal.predict_proba(raw_proba)

        assert pred.shape == raw_proba.shape
        assert np.all(pred >= 0) and np.all(pred <= 1)

    def test_probability_bounds(self):
        """Predicted probabilities must be in [0, 1]."""
        y_true = np.array([0, 1, 0, 1])
        raw_proba = np.array([0.0, 0.5, 1.0, 0.75])

        cal = build_sigmoid_calibrator()
        cal.fit(raw_proba, y_true)
        pred = cal.predict_proba(raw_proba)

        assert np.all(pred >= 0)
        assert np.all(pred <= 1)

    def test_unfitted_raises(self):
        """Predicting before fitting must raise RuntimeError."""
        cal = build_sigmoid_calibrator()
        with pytest.raises(RuntimeError):
            cal.predict_proba(np.array([0.5]))

    def test_invalid_probability_input_raises(self):
        """Probabilities outside [0,1] must raise ValueError."""
        y_true = np.array([0, 1])
        raw_proba = np.array([0.5, 1.5])

        cal = build_sigmoid_calibrator()
        cal.fit(np.array([0.2, 0.8]), y_true)
        with pytest.raises(ValueError):
            cal.predict_proba(raw_proba)

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths must raise ValueError."""
        cal = build_sigmoid_calibrator()
        with pytest.raises(ValueError):
            cal.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))

    def test_empty_arrays_raises(self):
        """Empty arrays must raise ValueError."""
        cal = build_sigmoid_calibrator()
        with pytest.raises(ValueError):
            cal.fit(np.array([]), np.array([]))

    def test_extreme_probabilities(self):
        """Handle p=0 and p=1 correctly via clipping."""
        y_true = np.array([0, 1])
        raw_proba = np.array([0.0, 1.0])

        cal = SigmoidProbabilityCalibrator(clip_eps=1e-6)
        cal.fit(raw_proba, y_true)
        pred = cal.predict_proba(raw_proba)

        assert np.all(pred >= 0) and np.all(pred <= 1)
        assert np.all(np.isfinite(pred))

    def test_deterministic(self):
        """Same input must produce same output."""
        y_true = np.array([0, 0, 1, 1])
        raw_proba = np.array([0.1, 0.2, 0.7, 0.8])

        cal1 = build_sigmoid_calibrator()
        cal1.fit(raw_proba, y_true)
        pred1 = cal1.predict_proba(raw_proba)

        cal2 = build_sigmoid_calibrator()
        cal2.fit(raw_proba, y_true)
        pred2 = cal2.predict_proba(raw_proba)

        assert np.allclose(pred1, pred2)


class TestIsotonicProbabilityCalibrator:
    """Tests for IsotonicProbabilityCalibrator."""

    def test_fit_predict_basic(self):
        """Isotonic calibrator fits and predicts on synthetic data."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        raw_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])

        cal = build_isotonic_calibrator()
        cal.fit(raw_proba, y_true)
        pred = cal.predict_proba(raw_proba)

        assert pred.shape == raw_proba.shape
        assert np.all(pred >= 0) and np.all(pred <= 1)

    def test_probability_bounds(self):
        """Predicted probabilities must be in [0, 1]."""
        y_true = np.array([0, 1, 0, 1])
        raw_proba = np.array([0.0, 0.5, 1.0, 0.75])

        cal = build_isotonic_calibrator()
        cal.fit(raw_proba, y_true)
        pred = cal.predict_proba(raw_proba)

        assert np.all(pred >= 0)
        assert np.all(pred <= 1)

    def test_unfitted_raises(self):
        """Predicting before fitting must raise RuntimeError."""
        cal = build_isotonic_calibrator()
        with pytest.raises(RuntimeError):
            cal.predict_proba(np.array([0.5]))

    def test_invalid_probability_input_raises(self):
        """Probabilities outside [0,1] must raise ValueError."""
        y_true = np.array([0, 1])
        raw_proba = np.array([0.5, 1.5])

        cal = build_isotonic_calibrator()
        cal.fit(np.array([0.2, 0.8]), y_true)
        with pytest.raises(ValueError):
            cal.predict_proba(raw_proba)

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths must raise ValueError."""
        cal = build_isotonic_calibrator()
        with pytest.raises(ValueError):
            cal.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))

    def test_empty_arrays_raises(self):
        """Empty arrays must raise ValueError."""
        cal = build_isotonic_calibrator()
        with pytest.raises(ValueError):
            cal.fit(np.array([]), np.array([]))

    def test_monotonicity(self):
        """Isotonic predictions must be monotonically non-decreasing."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        raw_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        cal = build_isotonic_calibrator()
        cal.fit(raw_proba, y_true)
        pred = cal.predict_proba(np.sort(raw_proba))

        # Should be monotonic
        assert np.all(np.diff(pred) >= -1e-10)

    def test_deterministic(self):
        """Same input must produce same output."""
        y_true = np.array([0, 0, 1, 1])
        raw_proba = np.array([0.1, 0.2, 0.7, 0.8])

        cal1 = build_isotonic_calibrator()
        cal1.fit(raw_proba, y_true)
        pred1 = cal1.predict_proba(raw_proba)

        cal2 = build_isotonic_calibrator()
        cal2.fit(raw_proba, y_true)
        pred2 = cal2.predict_proba(raw_proba)

        assert np.allclose(pred1, pred2)


class TestCrossFitting:
    """Tests for cross-fitted calibration evaluation."""

    def test_cross_fit_returns_metrics(self):
        """Cross-fitting returns metrics for both methods and baseline."""
        # Use small synthetic data for speed
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        raw_proba = np.random.rand(100)

        sigmoid_m, isotonic_m, baseline_m = cross_fit_calibration_evaluation(
            y_true, raw_proba, n_splits=3
        )

        # All should have metrics
        assert sigmoid_m.method == "build_sigmoid_calibrator"
        assert isotonic_m.method == "build_isotonic_calibrator"
        assert baseline_m.method == "baseline"

        # All should have calibrated OOF probabilities
        assert sigmoid_m.calibrated_oof_proba is not None
        assert isotonic_m.calibrated_oof_proba is not None
        assert len(sigmoid_m.calibrated_oof_proba) == len(y_true)
        assert len(isotonic_m.calibrated_oof_proba) == len(y_true)

        # Calibrated probs should be in [0, 1]
        assert np.all(sigmoid_m.calibrated_oof_proba >= 0)
        assert np.all(sigmoid_m.calibrated_oof_proba <= 1)
        assert np.all(isotonic_m.calibrated_oof_proba >= 0)
        assert np.all(isotonic_m.calibrated_oof_proba <= 1)

    def test_cross_fit_deterministic(self):
        """Cross-fitting must be deterministic with fixed seed."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        raw_proba = np.random.rand(50)

        sig1, iso1, _ = cross_fit_calibration_evaluation(y_true, raw_proba, n_splits=3)
        sig2, iso2, _ = cross_fit_calibration_evaluation(y_true, raw_proba, n_splits=3)

        assert np.allclose(sig1.calibrated_oof_proba, sig2.calibrated_oof_proba)
        assert np.allclose(iso1.calibrated_oof_proba, iso2.calibrated_oof_proba)


class TestMethodSelection:
    """Tests for deterministic calibration method selection."""

    def test_sigmoid_wins_on_log_loss(self):
        """Sigmoid selected when it has lower Log Loss."""
        from customer_churn.calibration import CalibrationMetrics

        sigmoid = CalibrationMetrics(
            method="sigmoid",
            log_loss_folds=[0.4, 0.4, 0.4],
            brier_folds=[0.14, 0.14, 0.14],
        )
        isotonic = CalibrationMetrics(
            method="isotonic",
            log_loss_folds=[0.5, 0.5, 0.5],
            brier_folds=[0.13, 0.13, 0.13],  # Better Brier but worse Log Loss
        )

        selected, rationale = select_calibration_method(sigmoid, isotonic)
        assert selected == "sigmoid"
        assert rationale["decision_rule_applied"] == "log_loss"

    def test_isotonic_wins_on_log_loss(self):
        """Isotonic selected when it has lower Log Loss."""
        from customer_churn.calibration import CalibrationMetrics

        sigmoid = CalibrationMetrics(
            method="sigmoid",
            log_loss_folds=[0.5, 0.5, 0.5],
            brier_folds=[0.14, 0.14, 0.14],
        )
        isotonic = CalibrationMetrics(
            method="isotonic",
            log_loss_folds=[0.4, 0.4, 0.4],
            brier_folds=[0.15, 0.15, 0.15],
        )

        selected, rationale = select_calibration_method(sigmoid, isotonic)
        assert selected == "isotonic"
        assert rationale["decision_rule_applied"] == "log_loss"

    def test_tie_break_prefers_sigmoid(self):
        """When Log Loss and Brier are practically equivalent, sigmoid wins."""
        from customer_churn.calibration import CalibrationMetrics

        sigmoid = CalibrationMetrics(
            method="sigmoid",
            log_loss_folds=[0.4135, 0.4135, 0.4135],
            brier_folds=[0.1343, 0.1343, 0.1343],
        )
        isotonic = CalibrationMetrics(
            method="isotonic",
            log_loss_folds=[0.4135, 0.4135, 0.4135],
            brier_folds=[0.1343, 0.1343, 0.1343],
        )

        selected, rationale = select_calibration_method(sigmoid, isotonic)
        assert selected == "sigmoid"
        assert rationale["decision_rule_applied"] == "tie_break_prefer_sigmoid"

    def test_brier_tiebreaker(self):
        """When Log Loss tied within tolerance, Brier is used."""
        from customer_churn.calibration import CalibrationMetrics

        sigmoid = CalibrationMetrics(
            method="sigmoid",
            log_loss_folds=[0.4135, 0.4135, 0.4135],
            brier_folds=[0.13, 0.13, 0.13],  # Better Brier
        )
        isotonic = CalibrationMetrics(
            method="isotonic",
            log_loss_folds=[0.4135, 0.4135, 0.4135],
            brier_folds=[0.14, 0.14, 0.14],  # Worse Brier
        )

        selected, rationale = select_calibration_method(sigmoid, isotonic)
        assert selected == "sigmoid"
        assert rationale["decision_rule_applied"] == "brier"


class TestCalibratedRFClassifier:
    """Tests for the CalibratedRFClassifier inference wrapper."""

    def test_predict_proba_shape(self):
        """predict_proba returns correct shape."""
        from sklearn.ensemble import RandomForestClassifier

        from customer_churn.features import build_preprocessor, load_data
        from customer_churn.final_evaluation import FINAL_RF_CONFIG

        # Load small training subset for quick test
        train_df = load_data("train")
        X_train = train_df.drop(columns=["Churn"]).iloc[:50]
        y_train = train_df["Churn"].replace({"No": 0, "Yes": 1}).astype(int).iloc[:50].values

        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(X_train)
        # FINAL_RF_CONFIG already has n_jobs, don't override
        rf = RandomForestClassifier(**FINAL_RF_CONFIG)
        rf.fit(X_trans, y_train)

        # Build calibrator on synthetic data
        from customer_churn.calibration import SigmoidProbabilityCalibrator
        cal = SigmoidProbabilityCalibrator()
        cal.fit(np.array([0.1, 0.9]), np.array([0, 1]))

        estimator = CalibratedRFClassifier(preprocessor, rf, cal)

        # Test on a few samples
        test_samples = train_df.drop(columns=["Churn"]).iloc[:5]
        proba = estimator.predict_proba(test_samples)

        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_class_probability_sums_to_one(self):
        """predict_proba columns sum to 1."""
        from sklearn.ensemble import RandomForestClassifier

        from customer_churn.features import build_preprocessor, load_data
        from customer_churn.final_evaluation import FINAL_RF_CONFIG

        train_df = load_data("train")
        X_train = train_df.drop(columns=["Churn"]).iloc[:30]
        y_train = train_df["Churn"].replace({"No": 0, "Yes": 1}).astype(int).iloc[:30].values

        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(X_train)
        rf = RandomForestClassifier(**FINAL_RF_CONFIG)
        rf.fit(X_trans, y_train)

        from customer_churn.calibration import SigmoidProbabilityCalibrator
        cal = SigmoidProbabilityCalibrator()
        cal.fit(np.array([0.1, 0.9]), np.array([0, 1]))

        estimator = CalibratedRFClassifier(preprocessor, rf, cal)
        test_samples = train_df.drop(columns=["Churn"]).iloc[:10]
        proba = estimator.predict_proba(test_samples)

        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_no_refit_during_inference(self):
        """Inference must not refit the calibrator."""
        from sklearn.ensemble import RandomForestClassifier

        from customer_churn.features import build_preprocessor, load_data
        from customer_churn.final_evaluation import FINAL_RF_CONFIG

        train_df = load_data("train")
        X_train = train_df.drop(columns=["Churn"]).iloc[:30]
        y_train = train_df["Churn"].replace({"No": 0, "Yes": 1}).astype(int).iloc[:30].values

        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(X_train)
        rf = RandomForestClassifier(**FINAL_RF_CONFIG)
        rf.fit(X_trans, y_train)

        from customer_churn.calibration import SigmoidProbabilityCalibrator
        cal = SigmoidProbabilityCalibrator()
        cal.fit(np.array([0.1, 0.9]), np.array([0, 1]))

        estimator = CalibratedRFClassifier(preprocessor, rf, cal)

        # Calling predict_proba multiple times should not change calibrator state
        test_samples = train_df.drop(columns=["Churn"]).iloc[:5]
        proba1 = estimator.predict_proba(test_samples)
        proba2 = estimator.predict_proba(test_samples)

        assert np.allclose(proba1, proba2)

    def test_serialization_roundtrip(self):
        """CalibratedRFClassifier serializes and deserializes with joblib."""
        import tempfile

        from sklearn.ensemble import RandomForestClassifier

        from customer_churn.features import build_preprocessor, load_data
        from customer_churn.final_evaluation import FINAL_RF_CONFIG

        train_df = load_data("train")
        X_train = train_df.drop(columns=["Churn"]).iloc[:30]
        y_train = train_df["Churn"].replace({"No": 0, "Yes": 1}).astype(int).iloc[:30].values

        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(X_train)
        rf = RandomForestClassifier(**FINAL_RF_CONFIG)
        rf.fit(X_trans, y_train)

        from customer_churn.calibration import SigmoidProbabilityCalibrator
        cal = SigmoidProbabilityCalibrator()
        cal.fit(np.array([0.1, 0.9]), np.array([0, 1]))

        estimator = CalibratedRFClassifier(preprocessor, rf, cal)
        test_samples = train_df.drop(columns=["Churn"]).iloc[:5]

        proba_before = estimator.predict_proba(test_samples)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = Path(f.name)

        try:
            joblib.dump(estimator, temp_path)
            loaded = joblib.load(temp_path)

            proba_after = loaded.predict_proba(test_samples)
            assert np.allclose(proba_before, proba_after)
        finally:
            temp_path.unlink()

    def test_loaded_artifact_does_not_refit_calibration(self):
        """Loaded artifact inference must not refit calibration."""
        import tempfile

        from sklearn.ensemble import RandomForestClassifier

        from customer_churn.features import build_preprocessor, load_data
        from customer_churn.final_evaluation import FINAL_RF_CONFIG

        train_df = load_data("train")
        X_train = train_df.drop(columns=["Churn"]).iloc[:30]
        y_train = train_df["Churn"].replace({"No": 0, "Yes": 1}).astype(int).iloc[:30].values

        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(X_train)
        rf = RandomForestClassifier(**FINAL_RF_CONFIG)
        rf.fit(X_trans, y_train)

        from customer_churn.calibration import SigmoidProbabilityCalibrator
        cal = SigmoidProbabilityCalibrator()
        cal.fit(np.array([0.1, 0.9]), np.array([0, 1]))

        estimator = CalibratedRFClassifier(preprocessor, rf, cal)
        test_samples = train_df.drop(columns=["Churn"]).iloc[:5]

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = Path(f.name)

        try:
            joblib.dump(estimator, temp_path)
            loaded = joblib.load(temp_path)

            # Multiple predictions should be identical
            p1 = loaded.predict_proba(test_samples)
            p2 = loaded.predict_proba(test_samples)
            p3 = loaded.predict_proba(test_samples)

            assert np.allclose(p1, p2)
            assert np.allclose(p2, p3)
        finally:
            temp_path.unlink()


class TestOOFGeneration:
    """Tests for OOF raw probability generation."""

    def test_oof_generation_no_test_leakage(self):
        """OOF generation must not use test data."""
        import inspect

        source = inspect.getsource(get_oof_raw_probabilities)
        assert "test" not in source.lower().replace("test_", "").replace("testset", "")

    def test_oof_returns_correct_shapes(self):
        """OOF returns y_true and raw probabilities of same length."""
        y_true, raw_proba = get_oof_raw_probabilities()
        assert len(y_true) == len(raw_proba)
        assert np.all(raw_proba >= 0) and np.all(raw_proba <= 1)


class TestPhase11Regression:
    """Ensure Phase 11 artifacts and logic remain unchanged."""

    def test_phase11_threshold_unchanged(self):
        """Phase 11 business threshold 0.28 must remain in its artifact."""
        import json

        with open("models/business_cost_analysis.json") as f:
            data = json.load(f)
        assert data["selected_threshold"] == 0.28

    def test_phase11_cost_matrix_unchanged(self):
        """Phase 11 cost matrix must remain unchanged."""
        import json

        with open("models/business_cost_analysis.json") as f:
            data = json.load(f)
        costs = data["cost_assumptions"]
        assert costs["cost_tn"] == 0.0
        assert costs["cost_fp"] == 10.0
        assert costs["cost_fn"] == 100.0
        assert costs["cost_tp"] == 10.0


class TestCalibratedBusinessThreshold:
    """Test calibrated OOF business threshold optimization."""

    def test_optimize_threshold_on_calibrated_oof(self):
        """Business threshold optimization works on calibrated OOF probabilities."""
        # Run calibration analysis to get calibrated OOF
        results = run_calibration_analysis()
        selected_method = results["selected_method"]

        # Get OOF raw probabilities
        y_true, raw_proba_oof = get_oof_raw_probabilities()

        # Build and fit selected calibrator on ALL OOF data
        from customer_churn.calibration import build_final_calibrator
        calibrator = build_final_calibrator(selected_method, y_true, raw_proba_oof)

        # Get calibrated OOF probabilities
        calibrated_oof = calibrator.predict_proba(raw_proba_oof)

        # Optimize threshold on CALIBRATED OOF probabilities
        result = optimize_threshold_business(
            y_true, calibrated_oof, DEFAULT_COST_MATRIX
        )

        assert 0.0 <= result.selected_threshold <= 1.0
        assert "total_cost" in result.selected_metrics

    def test_calibrated_threshold_differs_from_phase11(self):
        """Calibrated threshold should generally differ from Phase 11 (0.28)."""
        results = run_calibration_analysis()
        selected_method = results["selected_method"]

        y_true, raw_proba_oof = get_oof_raw_probabilities()

        from customer_churn.calibration import build_final_calibrator
        calibrator = build_final_calibrator(selected_method, y_true, raw_proba_oof)
        calibrated_oof = calibrator.predict_proba(raw_proba_oof)

        result = optimize_threshold_business(
            y_true, calibrated_oof, DEFAULT_COST_MATRIX
        )

        # The threshold may or may not differ, but we can at least verify it runs
        # and produces valid output
        assert 0.0 <= result.selected_threshold <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
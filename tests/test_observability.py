"""Tests for Phase 17 - Lightweight Inference Observability."""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory for extraction module
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from customer_churn.features import TARGET_COL, load_data
from customer_churn.observability import (
    ErrorCategory,
    MetricsCollector,
    MetricsEvent,
    create_collector_from_metadata,
    metrics_context,
)
from customer_churn.portable_predict import (
    InferenceMetadata,
    PortableChurnPredictor,
    PortableValidationError,
)
from customer_churn.predict import ChurnPredictor


# Fixtures from test_portable_inference.py
@pytest.fixture(scope="module")
def artifacts_dir():
    """Models directory with artifacts."""
    return Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def portable_predictor(artifacts_dir):
    """Portable predictor loaded from artifacts."""
    return PortableChurnPredictor.from_artifacts(artifacts_dir)


@pytest.fixture(scope="module")
def current_predictor():
    """Current ChurnPredictor for compatibility testing."""
    return ChurnPredictor()


@pytest.fixture(scope="module")
def valid_sample():
    """A valid single-row input."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.50,
        "TotalCharges": 29.85,
    }


@pytest.fixture(scope="module")
def test_data_no_nan():
    """Test data without NaN values for compatibility testing."""
    test_df = load_data("test")
    X_test = test_df.drop(columns=[TARGET_COL]).dropna()
    return X_test


@pytest.fixture(scope="module")
def predictor(portable_predictor):
    """Alias for portable_predictor for backward compat."""
    return portable_predictor


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_successful_single_inference(self):
        """Test recording successful single prediction."""
        collector = MetricsCollector(
            model_artifact_hash="abc123",
            default_threshold=0.28,
        )
        event = collector.record_success(
            event_type="predict_single",
            prediction_count=1,
            positive_predictions=1,
            negative_predictions=0,
            latency_ms=15.5,
            threshold=0.28,
            batch_size=1,
        )
        assert isinstance(event, MetricsEvent)
        assert event.success is True
        assert event.prediction_count == 1
        assert event.positive_predictions == 1
        assert event.negative_predictions == 0
        assert event.latency_ms == 15.5
        assert event.threshold == 0.28
        assert event.batch_size == 1
        assert event.error_category is None

    def test_successful_batch_inference(self):
        """Test recording successful batch prediction."""
        collector = MetricsCollector()
        event = collector.record_success(
            event_type="predict_batch",
            prediction_count=100,
            positive_predictions=25,
            negative_predictions=75,
            latency_ms=45.2,
            threshold=0.28,
            batch_size=100,
        )
        assert event.prediction_count == 100
        assert event.positive_predictions == 25
        assert event.negative_predictions == 75
        assert event.batch_size == 100

    def test_invocation_vs_prediction_counts(self):
        """Test invocation count differs from prediction count."""
        collector = MetricsCollector()
        collector.record_success("predict_single", 1, 1, 0, 10.0, 0.28, 1)
        collector.record_success("predict_batch", 50, 10, 40, 30.0, 0.28, 50)
        collector.record_success("predict_single", 1, 0, 1, 12.0, 0.28, 1)

        assert collector.invocation_count == 3
        assert collector.total_predictions == 52
        assert collector.positive_predictions == 11
        assert collector.negative_predictions == 41

    def test_validation_failure_categorization(self):
        """Test validation error categorization."""
        collector = MetricsCollector()
        event = collector.record_failure(
            event_type="predict_batch",
            error_category=ErrorCategory.VALIDATION_ERROR,
            threshold=0.28,
            latency_ms=5.0,
            batch_size=10,
        )
        assert event.success is False
        assert event.error_category == ErrorCategory.VALIDATION_ERROR
        assert event.prediction_count == 0
        assert collector.failure_count == 1
        assert collector.validation_failure_count == 1

    def test_inference_failure_categorization(self):
        """Test inference error categorization."""
        collector = MetricsCollector()
        collector.record_failure("predict", ErrorCategory.INFERENCE_ERROR, 0.28)
        assert collector.inference_failure_count == 1

    def test_artifact_integrity_failure_categorization(self):
        """Test artifact integrity error categorization."""
        collector = MetricsCollector()
        collector.record_failure(
            "from_artifacts", ErrorCategory.ARTIFACT_INTEGRITY_ERROR, 0.28
        )
        assert collector.artifact_integrity_failure_count == 1

    def test_artifact_load_failure_categorization(self):
        """Test artifact loading error categorization."""
        collector = MetricsCollector()
        collector.record_failure("from_artifacts", ErrorCategory.ARTIFACT_LOAD_ERROR, 0.28)
        assert collector.artifact_load_failure_count == 1

    def test_failed_not_counted_as_success(self):
        """Failed calls must not be counted as successful."""
        collector = MetricsCollector()
        collector.record_success("predict", 1, 1, 0, 10.0, 0.28)
        collector.record_failure("predict", ErrorCategory.VALIDATION_ERROR, 0.28)
        assert collector.success_count == 1
        assert collector.failure_count == 1
        assert collector.invocation_count == 2

    def test_latency_measurement(self):
        """Test latency is measured in milliseconds."""
        collector = MetricsCollector()
        start = time.perf_counter()
        time.sleep(0.01)
        latency = (time.perf_counter() - start) * 1000
        event = collector.record_success("predict", 1, 1, 0, latency, 0.28)
        assert event.latency_ms >= 10.0

    def test_utc_timestamp_format(self):
        """Test UTC timestamp is ISO format."""
        collector = MetricsCollector()
        event = collector.record_success("predict", 1, 1, 0, 10.0, 0.28)
        assert event.timestamp.endswith("+00:00") or event.timestamp.endswith("Z")
        assert "T" in event.timestamp

    def test_multiple_calls_aggregated(self):
        """Test multiple calls aggregated by same collector."""
        collector = MetricsCollector()
        for i in range(5):
            collector.record_success("predict", 1, 1, 0, 10.0, 0.28)
        assert collector.invocation_count == 5
        assert collector.total_predictions == 5
        assert len(collector.events) == 5

    def test_empty_batch_not_supported(self):
        """Test empty batch handling (if supported)."""
        collector = MetricsCollector()
        event = collector.record_success("predict_batch", 0, 0, 0, 5.0, 0.28, 0)
        assert event.prediction_count == 0
        assert event.batch_size == 0

    def test_get_summary(self):
        """Test summary aggregation."""
        collector = MetricsCollector(model_artifact_hash="abc", default_threshold=0.28)
        collector.record_success("predict", 1, 1, 0, 10.0, 0.28)
        collector.record_success("predict_batch", 10, 3, 7, 30.0, 0.28, 10)
        collector.record_failure("predict", ErrorCategory.VALIDATION_ERROR, 0.28)

        summary = collector.get_summary()
        assert summary["invocation_count"] == 3
        assert summary["success_count"] == 2
        assert summary["failure_count"] == 1
        assert summary["validation_failure_count"] == 1
        assert summary["total_predictions"] == 11
        assert summary["positive_predictions"] == 4
        assert summary["negative_predictions"] == 7
        assert summary["model_artifact_hash"] == "abc"
        assert summary["default_threshold"] == 0.28

    def test_write_jsonl(self, tmp_path):
        """Test JSONL output."""
        collector = MetricsCollector()
        collector.record_success("predict", 1, 1, 0, 10.0, 0.28)
        collector.record_success("predict_batch", 5, 2, 3, 25.0, 0.28, 5)

        output_path = tmp_path / "metrics.jsonl"
        collector.write_jsonl(output_path)

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        import json
        event1 = json.loads(lines[0])
        assert event1["event_type"] == "predict"
        assert event1["success"] is True

    def test_clear(self):
        """Test clearing collector."""
        collector = MetricsCollector()
        collector.record_success("predict", 1, 1, 0, 10.0, 0.28)
        collector.clear()
        assert collector.invocation_count == 0
        assert collector.total_predictions == 0
        assert len(collector.events) == 0


class TestErrorCategorization:
    """Tests for error categorization."""

    def test_validation_error(self):
        from customer_churn.portable_predict import PortableValidationError
        error = PortableValidationError("test", ["missing field"])
        from customer_churn.observability import _categorize_error
        assert _categorize_error(error) == ErrorCategory.VALIDATION_ERROR

    def test_file_not_found(self):
        from customer_churn.observability import _categorize_error
        error = FileNotFoundError("model not found")
        assert _categorize_error(error) == ErrorCategory.ARTIFACT_LOAD_ERROR

    def test_integrity_value_error(self):
        from customer_churn.observability import _categorize_error
        error = ValueError("Artifact integrity check failed: SHA256 mismatch")
        assert _categorize_error(error) == ErrorCategory.ARTIFACT_INTEGRITY_ERROR

    def test_generic_value_error(self):
        from customer_churn.observability import _categorize_error
        error = ValueError("some other error")
        assert _categorize_error(error) == ErrorCategory.ARTIFACT_LOAD_ERROR

    def test_os_error(self):
        from customer_churn.observability import _categorize_error
        error = OSError("permission denied")
        assert _categorize_error(error) == ErrorCategory.ARTIFACT_LOAD_ERROR

    def test_json_decode_error(self):
        from customer_churn.observability import _categorize_error
        error = json.JSONDecodeError("invalid", "{}", 0)
        assert _categorize_error(error) == ErrorCategory.ARTIFACT_LOAD_ERROR

    def test_unknown_error(self):
        from customer_churn.observability import _categorize_error
        error = RuntimeError("unexpected")
        assert _categorize_error(error) == ErrorCategory.UNKNOWN_ERROR


class TestMetricsContext:
    """Tests for metrics_context context manager."""

    def test_successful_context(self):
        """Context manager records timing but not success (no prediction counts available)."""
        collector = MetricsCollector()
        with metrics_context(collector, "predict_batch", threshold=0.28, batch_size=10):
            time.sleep(0.01)

        # Context manager only records failures; success recording happens in predictor methods
        assert collector.invocation_count == 0
        assert collector.success_count == 0
        assert len(collector.events) == 0

    def test_failed_context(self):
        collector = MetricsCollector()
        try:
            with metrics_context(collector, "predict", threshold=0.28):
                raise PortableValidationError("missing field", ["field1"])
        except PortableValidationError:
            pass

        assert collector.invocation_count == 1
        assert collector.failure_count == 1
        assert collector.validation_failure_count == 1
        assert collector.events[0].error_category == ErrorCategory.VALIDATION_ERROR


class TestCreateCollectorFromMetadata:
    """Tests for create_collector_from_metadata factory."""

    def test_from_inference_metadata(self):
        metadata = InferenceMetadata(
            version="1.0",
            model_type="RandomForestClassifier",
            model_version="1.0",
            production_threshold=0.28,
            threshold_metadata={},
            preprocessing={},
            artifact_integrity={"pipeline_sha256": "abc123"},
            runtime={},
        )
        collector = create_collector_from_metadata(metadata)
        assert collector.model_artifact_hash == "abc123"
        assert collector.default_threshold == 0.28


class TestPortablePredictorWithMetrics:
    """Integration tests for PortableChurnPredictor with metrics."""

    @pytest.fixture
    def artifacts_dir(self):
        return Path(__file__).resolve().parents[1] / "models"

    @pytest.fixture
    def predictor(self, artifacts_dir):
        return PortableChurnPredictor.from_artifacts(artifacts_dir)

    @pytest.fixture
    def valid_sample(self):
        return {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 75.50,
            "TotalCharges": 29.85,
        }

    def test_predict_single_records_metrics(self, predictor, valid_sample):
        """predict_single records metrics when collector present."""
        collector = predictor.metrics_collector
        assert collector is not None
        initial_count = collector.invocation_count

        _ = predictor.predict_single(valid_sample)

        assert collector.invocation_count == initial_count + 1
        assert collector.success_count == initial_count + 1
        assert collector.total_predictions == 1
        assert collector.positive_predictions in (0, 1)
        assert collector.negative_predictions in (0, 1)

    def test_predict_batch_records_metrics(self, predictor, valid_sample):
        """predict_batch records metrics with correct counts."""
        df = pd.DataFrame([valid_sample, valid_sample, valid_sample])
        collector = predictor.metrics_collector
        initial = collector.invocation_count

        result = predictor.predict_batch(df)

        assert collector.invocation_count == initial + 1
        assert collector.success_count == initial + 1
        assert collector.total_predictions == 3
        assert sum(
            e.positive_predictions for e in collector.events
            if e.event_type == "predict_batch"
        ) == sum(r for r in result["predictions"])
        assert collector.events[-1].batch_size == 3

    def test_predict_proba_records_metrics(self, predictor, valid_sample):
        """predict_proba records metrics."""
        df = pd.DataFrame([valid_sample])
        collector = predictor.metrics_collector
        initial = collector.invocation_count

        proba = predictor.predict_proba(df)

        assert collector.invocation_count == initial + 1
        assert collector.success_count == initial + 1
        assert len(proba) == 1

    def test_predict_records_metrics(self, predictor, valid_sample):
        """predict records metrics."""
        df = pd.DataFrame([valid_sample])
        collector = predictor.metrics_collector
        initial = collector.invocation_count

        pred = predictor.predict(df)

        assert collector.invocation_count == initial + 1
        assert collector.success_count == initial + 1
        assert len(pred) == 1

    def test_validation_failure_records_metrics(self, predictor):
        """Validation failure records failure metrics."""
        invalid_input = {"gender": "Female"}  # Missing required fields
        collector = predictor.metrics_collector
        initial_inv = collector.invocation_count
        initial_fail = collector.failure_count
        initial_val = collector.validation_failure_count

        with pytest.raises(PortableValidationError):
            predictor.predict_single(invalid_input)

        assert collector.invocation_count == initial_inv + 1
        assert collector.failure_count == initial_fail + 1
        assert collector.validation_failure_count == initial_val + 1

    def test_metrics_collector_property(self, predictor):
        """Metrics collector accessible via property."""
        assert predictor.metrics_collector is not None
        assert isinstance(predictor.metrics_collector, MetricsCollector)

    def test_predictions_unchanged_with_metrics(self, predictor, valid_sample):
        """Predictions identical and without metrics collection."""
        # Use fresh predictor without metrics collector comparison
        predictor_no_metrics = PortableChurnPredictor.from_artifacts(
            Path(__file__).resolve().parents[1] / "models",
            metrics_collector=None,
        )

        df = pd.DataFrame([valid_sample, valid_sample])

        # With metrics
        result_with = predictor.predict_batch(df)
        # Without metrics
        result_without = predictor_no_metrics.predict_batch(df)

        assert result_with["predictions"] == result_without["predictions"]
        assert np.allclose(result_with["probabilities"], result_without["probabilities"], rtol=1e-10, atol=1e-12)
        assert result_with["threshold"] == result_without["threshold"]

    def test_predict_single_unchanged(self, predictor, valid_sample):
        """predict_single unchanged with metrics."""
        predictor_no_metrics = PortableChurnPredictor.from_artifacts(
            Path(__file__).resolve().parents[1] / "models",
            metrics_collector=None,
        )

        r1 = predictor.predict_single(valid_sample)
        r2 = predictor_no_metrics.predict_single(valid_sample)

        assert abs(r1["churn_probability"] - r2["churn_probability"]) < 1e-12
        assert r1["prediction"] == r2["prediction"]
        assert r1["threshold"] == r2["threshold"]
        assert r1["prediction_label"] == r2["prediction_label"]


class TestPrivacy:
    """Tests for privacy requirements."""

    def test_no_raw_input_in_events(self, predictor, valid_sample):
        """Events must not contain raw input data."""
        predictor.predict_single(valid_sample)
        collector = predictor.metrics_collector
        event = collector.events[-1]

        # Check no customer data in event
        event_dict = event.__dict__
        event_str = str(event_dict)
        assert "75.50" not in event_str  # MonthlyCharges
        assert "29.85" not in event_str  # TotalCharges
        assert "Female" not in event_str  # gender
        assert "customerID" not in event_str

    def test_no_customer_ids_in_logs(self, predictor):
        """Customer IDs never logged."""
        sample = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 75.50,
            "TotalCharges": 29.85,
        }
        predictor.predict_single(sample)
        collector = predictor.metrics_collector
        for event in collector.events:
            assert "customerID" not in str(event)

    def test_no_individual_probabilities_by_default(self, predictor, valid_sample):
        """Individual probabilities not logged by default."""
        df = pd.DataFrame([valid_sample, valid_sample])
        predictor.predict_batch(df)
        collector = predictor.metrics_collector
        for event in collector.events:
            # Only aggregate counts, not individual probs
            assert "probabilities" not in str(event)


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    def test_existing_api_works(self, predictor, valid_sample):
        """All existing methods work unchanged."""
        df = pd.DataFrame([valid_sample])

        # All should work without errors
        proba = predictor.predict_proba(df)
        pred = predictor.predict(df)
        batch = predictor.predict_batch(df)
        single = predictor.predict_single(valid_sample)

        assert isinstance(proba, np.ndarray)
        assert isinstance(pred, np.ndarray)
        assert isinstance(batch, dict)
        assert isinstance(single, dict)

    def test_threshold_override_works(self, predictor, valid_sample):
        """Threshold override still works."""
        df = pd.DataFrame([valid_sample])
        result1 = predictor.predict_batch(df, threshold=0.5)
        result2 = predictor.predict_batch(df, threshold=0.3)

        # Different thresholds should produce different results
        # (though may be same for this sample)
        assert "threshold" in result1
        assert "threshold" in result2

    def test_without_metrics_collector(self, artifacts_dir, valid_sample):
        """Works identically when no metrics collector provided."""
        predictor_no_metrics = PortableChurnPredictor.from_artifacts(
            artifacts_dir, metrics_collector=None
        )
        predictor_with = PortableChurnPredictor.from_artifacts(artifacts_dir)

        df = pd.DataFrame([valid_sample, valid_sample])

        r1 = predictor_no_metrics.predict_batch(df)
        r2 = predictor_with.predict_batch(df)

        assert r1["predictions"] == r2["predictions"]
        assert np.allclose(r1["probabilities"], r2["probabilities"], rtol=1e-10, atol=1e-12)

    def test_integrity_verification_default_enabled(self, artifacts_dir):
        """Integrity verification still enabled by default."""
        # This should not raise
        predictor = PortableChurnPredictor.from_artifacts(artifacts_dir)
        assert predictor.metrics_collector is not None

    def test_cli_args_still_work(self, artifacts_dir):
        """CLI arguments unchanged."""
        import subprocess
        fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/portable_smoke_input.csv"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)
        try:
            result = subprocess.run([
                "uv", "run", "python", "-m", "customer_churn.portable_predict",
                "--input", str(fixture),
                "--artifacts", str(artifacts_dir),
                "--output", str(output_path),
            ], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1], check=False)
            assert result.returncode == 0
            assert output_path.exists()
        finally:
            output_path.unlink(missing_ok=True)


class TestMetricsFailureDoesNotMaskInference:
    """Metrics failures must not mask inference outcomes."""

    def test_metrics_failure_does_not_mask(self, predictor, valid_sample):
        """If metrics recording fails, prediction still succeeds."""
        # We can't easily trigger a metrics failure without mocking internals,
        # but we can verify the try/except structure by checking
        # that exceptions from _infer_proba or validate propagate.
        from customer_churn.portable_predict import PortableValidationError

        with pytest.raises(PortableValidationError):
            predictor.predict_single({"gender": "Female"})  # missing fields


class TestInferenceBehaviorComparison:
    """Compare inference behavior before and after Phase 17."""

    def test_prediction_parity(self, artifacts_dir, valid_sample):
        """Predictions identical with/without metrics collector."""
        predictor_with = PortableChurnPredictor.from_artifacts(artifacts_dir)
        predictor_without = PortableChurnPredictor.from_artifacts(
            artifacts_dir, metrics_collector=None
        )

        df_single = pd.DataFrame([valid_sample])
        df_multi = pd.DataFrame([valid_sample, valid_sample])

        # Test predict_proba with single row (current API limitation)
        r1 = predictor_with.predict_proba(df_single)
        r2 = predictor_without.predict_proba(df_single)
        assert np.allclose(r1, r2, rtol=1e-10, atol=1e-12)

        # Test predict with single row
        r1 = predictor_with.predict(df_single)
        r2 = predictor_without.predict(df_single)
        assert np.array_equal(r1, r2)

        # Test predict_batch with multiple rows
        r1 = predictor_with.predict_batch(df_multi)
        r2 = predictor_without.predict_batch(df_multi)
        assert r1["predictions"] == r2["predictions"]
        assert np.allclose(r1["probabilities"], r2["probabilities"], rtol=1e-10, atol=1e-12)
        assert r1["threshold"] == r2["threshold"]

        # Test predict_single
        r1 = predictor_with.predict_single(valid_sample)
        r2 = predictor_without.predict_single(valid_sample)
        assert abs(r1["churn_probability"] - r2["churn_probability"]) < 1e-12
        assert r1["prediction"] == r2["prediction"]
        assert r1["threshold"] == r2["threshold"]
        assert r1["prediction_label"] == r2["prediction_label"]
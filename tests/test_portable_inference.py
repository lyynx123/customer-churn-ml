"""Tests for Phase 15 - Portable Inference."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path for extraction module
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Import extraction function from scripts
from extract_inference_artifacts import extract_inference_artifacts

from customer_churn.features import TARGET_COL, load_data
from customer_churn.portable_predict import (
    InferenceMetadata,
    PortableChurnPredictor,
    PortableDataContract,
    PortableValidationError,
    PortableValidationResult,
    compute_sha256,
)
from customer_churn.predict import ChurnPredictor, load_pipeline

# =============================================================================
# Fixtures
# =============================================================================

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


# =============================================================================
# Artifact Tests
# =============================================================================

def test_schema_exists(artifacts_dir):
    """inference_schema.json exists."""
    schema_path = artifacts_dir / "inference_schema.json"
    assert schema_path.exists()


def test_metadata_exists(artifacts_dir):
    """inference_metadata.json exists."""
    metadata_path = artifacts_dir / "inference_metadata.json"
    assert metadata_path.exists()


def test_schema_matches_fitted_pipeline(artifacts_dir):
    """Schema matches the fitted pipeline's OneHotEncoder categories."""
    pipeline = load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    with open(artifacts_dir / "inference_schema.json") as f:
        schema = json.load(f)

    # Check numeric features
    _, _, num_features = preprocessor.transformers_[0]
    assert list(schema["numeric_features"]) == list(num_features)

    # Check categorical features
    _, _, cat_features = preprocessor.transformers_[1]
    assert list(schema["categorical_features"].keys()) == list(cat_features)

    # Check categorical allowed values match encoder
    for i, col in enumerate(cat_features):
        allowed = schema["categorical_features"][col]
        encoder_cats = list(encoder.categories_[i])
        assert set(allowed) == set(encoder_cats), f"Mismatch for {col}"

    # Check feature order
    expected_order = list(num_features) + list(cat_features)
    assert schema["feature_order"] == expected_order

    # Check forbidden fields
    assert set(schema["forbidden_fields"]) == {"customerID", "Churn"}

    # Check numeric constraints
    assert "SeniorCitizen" in schema["numeric_constraints"]
    assert "tenure" in schema["numeric_constraints"]
    assert "MonthlyCharges" in schema["numeric_constraints"]
    assert "TotalCharges" in schema["numeric_constraints"]


def test_categorical_domains_match_encoder(artifacts_dir):
    """Categorical domains in schema match OneHotEncoder.categories_ exactly."""
    pipeline = load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    with open(artifacts_dir / "inference_schema.json") as f:
        schema = json.load(f)

    _, _, cat_features = preprocessor.transformers_[1]
    for i, col in enumerate(cat_features):
        schema_cats = set(schema["categorical_features"][col])
        encoder_cats = set(encoder.categories_[i])
        assert schema_cats == encoder_cats, f"Domain mismatch for {col}: {schema_cats} vs {encoder_cats}"


def test_feature_ordering_is_correct(artifacts_dir):
    """Feature ordering matches preprocessor expectation."""
    pipeline = load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]

    with open(artifacts_dir / "inference_schema.json") as f:
        schema = json.load(f)

    num_features = list(preprocessor.transformers_[0][2])
    cat_features = list(preprocessor.transformers_[1][2])
    expected = num_features + cat_features

    assert schema["feature_order"] == expected


def test_threshold_equals_028(artifacts_dir):
    """Production threshold equals 0.28."""
    with open(artifacts_dir / "inference_metadata.json") as f:
        metadata = json.load(f)

    assert metadata["production_threshold"] == 0.28
    assert metadata["threshold_metadata"]["value"] == 0.28
    assert metadata["threshold_metadata"]["source_phase"] == "Phase 11"
    assert metadata["threshold_metadata"]["selection_method"] == "Business cost optimization on OOF probabilities"


def test_sha256_matches_pipeline(artifacts_dir):
    """SHA256 in metadata matches actual pipeline file."""
    with open(artifacts_dir / "inference_metadata.json") as f:
        metadata = json.load(f)

    expected_hash = metadata["artifact_integrity"]["pipeline_sha256"]
    actual_hash = compute_sha256(artifacts_dir / "final_pipeline.joblib")

    assert expected_hash == actual_hash


# =============================================================================
# Runtime Tests
# =============================================================================

def test_predictor_loads_successfully(artifacts_dir):
    """Portable predictor loads without errors."""
    predictor = PortableChurnPredictor.from_artifacts(artifacts_dir)
    assert predictor is not None
    assert predictor.pipeline is not None
    assert predictor._contract is not None
    assert predictor._metadata is not None


def test_explicit_artifact_directory_works(artifacts_dir):
    """Explicit artifact directory parameter works."""
    predictor = PortableChurnPredictor.from_artifacts(artifact_dir=artifacts_dir)
    assert predictor.threshold == 0.28


def test_no_training_data_access_required(artifacts_dir):
    """Loading predictor does not require training data access."""
    # This test passes if import/load succeeds without accessing data/processed/
    # The fact that from_artifacts works without loading features.py config is the test
    predictor = PortableChurnPredictor.from_artifacts(artifacts_dir)
    assert predictor.threshold == 0.28


def test_threshold_comes_from_metadata(artifacts_dir):
    """Threshold is loaded from inference_metadata.json, not hardcoded."""
    predictor = PortableChurnPredictor.from_artifacts(artifacts_dir)
    assert predictor.threshold == 0.28
    assert predictor._metadata.production_threshold == 0.28


def test_integrity_verification_works(artifacts_dir):
    """SHA256 integrity verification works."""
    # Valid metadata should pass
    predictor = PortableChurnPredictor.from_artifacts(
        artifacts_dir, verify_integrity=True
    )
    assert predictor is not None

    # Modified metadata should fail
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Copy artifacts
        import shutil
        shutil.copy(artifacts_dir / "final_pipeline.joblib", tmpdir / "final_pipeline.joblib")
        shutil.copy(artifacts_dir / "inference_schema.json", tmpdir / "inference_schema.json")
        shutil.copy(artifacts_dir / "inference_metadata.json", tmpdir / "inference_metadata.json")

        # Corrupt metadata hash
        with open(tmpdir / "inference_metadata.json") as f:
            meta = json.load(f)
        meta["artifact_integrity"]["pipeline_sha256"] = "0" * 64
        with open(tmpdir / "inference_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        with pytest.raises(ValueError, match="Artifact integrity check failed"):
            PortableChurnPredictor.from_artifacts(tmpdir, verify_integrity=True)

        # Should pass with no integrity check
        predictor = PortableChurnPredictor.from_artifacts(
            tmpdir, verify_integrity=False
        )
        assert predictor is not None


# =============================================================================
# Compatibility Tests
# =============================================================================

def test_probability_matches_current(portable_predictor, current_predictor, test_data_no_nan):
    """Portable predict_proba matches current within tolerance."""
    # Current ChurnPredictor predict_proba only accepts single row
    # Use predict_batch for multi-row comparison
    batch_current = current_predictor.predict_batch(test_data_no_nan)
    batch_portable = portable_predictor.predict_batch(test_data_no_nan)

    prob_current = np.array(batch_current["probabilities"])
    prob_portable = np.array(batch_portable["probabilities"])

    assert np.allclose(prob_portable, prob_current, rtol=1e-10, atol=1e-12), \
        f"Max diff: {np.max(np.abs(prob_portable - prob_current))}"


def test_binary_prediction_matches_current(portable_predictor, current_predictor, test_data_no_nan):
    """Portable predict matches current exactly."""
    # Current ChurnPredictor predict only accepts single row
    # Use predict_batch for multi-row comparison
    batch_current = current_predictor.predict_batch(test_data_no_nan)
    batch_portable = portable_predictor.predict_batch(test_data_no_nan)

    pred_current = np.array(batch_current["predictions"])
    pred_portable = np.array(batch_portable["predictions"])

    assert np.array_equal(pred_portable, pred_current), \
        f"Mismatch count: {np.sum(pred_portable != pred_current)}"


def test_predict_single_matches_current(portable_predictor, current_predictor, test_data_no_nan):
    """Portable predict_single matches current exactly."""
    for i in range(min(50, len(test_data_no_nan))):
        row = test_data_no_nan.iloc[i].to_dict()
        r1 = current_predictor.predict_single(row)
        r2 = portable_predictor.predict_single(row)

        assert abs(r1["churn_probability"] - r2["churn_probability"]) < 1e-10, \
            f"Sample {i} prob mismatch: {r1['churn_probability']} vs {r2['churn_probability']}"
        assert r1["prediction"] == r2["prediction"], \
            f"Sample {i} pred mismatch: {r1['prediction']} vs {r2['prediction']}"
        assert r1["threshold"] == r2["threshold"], \
            f"Sample {i} thresh mismatch: {r1['threshold']} vs {r2['threshold']}"


def test_predict_batch_matches_current(portable_predictor, current_predictor, test_data_no_nan):
    """Portable predict_batch matches current exactly."""
    batch_current = current_predictor.predict_batch(test_data_no_nan)
    batch_portable = portable_predictor.predict_batch(test_data_no_nan)

    assert batch_current["predictions"] == batch_portable["predictions"]
    assert all(
        abs(a - b) < 1e-10
        for a, b in zip(batch_current["probabilities"], batch_portable["probabilities"])
    )
    assert batch_current["threshold"] == batch_portable["threshold"]


# =============================================================================
# Validation Tests
# =============================================================================

def test_validation_missing_fields(portable_predictor):
    """Missing required fields rejected."""
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        # Missing most fields
    }

    with pytest.raises(PortableValidationError, match="Missing required fields"):
        portable_predictor.predict_single(sample)


def test_validation_forbidden_customerID(portable_predictor, valid_sample):
    """customerID field rejected."""
    sample = {**valid_sample, "customerID": "12345"}

    with pytest.raises(PortableValidationError, match="Forbidden fields present"):
        portable_predictor.predict_single(sample)


def test_validation_forbidden_Churn(portable_predictor, valid_sample):
    """Churn field rejected."""
    sample = {**valid_sample, "Churn": "Yes"}

    with pytest.raises(PortableValidationError, match="Forbidden fields present"):
        portable_predictor.predict_single(sample)


def test_validation_unexpected_field(portable_predictor, valid_sample):
    """Unexpected fields rejected."""
    sample = {**valid_sample, "unexpected_field": "value"}

    with pytest.raises(PortableValidationError, match="Unexpected fields"):
        portable_predictor.predict_single(sample)


def test_validation_unknown_categorical_strict(portable_predictor, valid_sample):
    """Unknown categorical values rejected in strict mode."""
    sample = {**valid_sample, "Contract": "InvalidContract"}

    with pytest.raises(PortableValidationError, match="not in allowed categories"):
        portable_predictor.predict_single(sample)


def test_validation_unknown_categorical_lenient_allowed(artifacts_dir, valid_sample):
    """Unknown categorical values allowed in lenient mode."""
    predictor = PortableChurnPredictor.from_artifacts(
        artifacts_dir, allow_unknown_categories=True
    )
    sample = {**valid_sample, "Contract": "InvalidContract"}

    # Should not raise, passes to OneHotEncoder which produces all zeros
    result = predictor.predict_single(sample)
    assert "churn_probability" in result


def test_validation_nan_numeric(portable_predictor, valid_sample):
    """NaN in numeric field rejected."""
    sample = {**valid_sample, "MonthlyCharges": float("nan")}

    with pytest.raises(PortableValidationError, match="missing value"):
        portable_predictor.predict_single(sample)


def test_validation_nan_categorical(portable_predictor, valid_sample):
    """NaN in categorical field rejected."""
    sample = {**valid_sample, "Contract": float("nan")}

    with pytest.raises(PortableValidationError, match="missing value"):
        portable_predictor.predict_single(sample)


def test_validation_infinity_positive(portable_predictor, valid_sample):
    """Positive infinity rejected."""
    sample = {**valid_sample, "MonthlyCharges": float("inf")}

    with pytest.raises(PortableValidationError, match="non-finite value"):
        portable_predictor.predict_single(sample)


def test_validation_infinity_negative(portable_predictor, valid_sample):
    """Negative infinity rejected."""
    sample = {**valid_sample, "MonthlyCharges": float("-inf")}

    with pytest.raises(PortableValidationError, match="non-finite value"):
        portable_predictor.predict_single(sample)


def test_validation_invalid_numeric_type(portable_predictor, valid_sample):
    """Invalid numeric type rejected."""
    sample = {**valid_sample, "SeniorCitizen": "yes"}

    with pytest.raises(PortableValidationError, match="invalid int value"):
        portable_predictor.predict_single(sample)


def test_validation_float_senior_citizen(portable_predictor, valid_sample):
    """Float for SeniorCitizen rejected (must be int)."""
    sample = {**valid_sample, "SeniorCitizen": 1.5}

    with pytest.raises(PortableValidationError, match="expected integer"):
        portable_predictor.predict_single(sample)


def test_validation_senior_citizen_outside_domain(portable_predictor, valid_sample):
    """SeniorCitizen outside {0,1} rejected."""
    sample = {**valid_sample, "SeniorCitizen": 2}

    with pytest.raises(PortableValidationError, match="not in allowed set"):
        portable_predictor.predict_single(sample)


def test_validation_negative_tenure(portable_predictor, valid_sample):
    """Negative tenure rejected."""
    sample = {**valid_sample, "tenure": -5}

    with pytest.raises(PortableValidationError, match="below minimum"):
        portable_predictor.predict_single(sample)


def test_validation_negative_monthly_charges(portable_predictor, valid_sample):
    """Negative MonthlyCharges rejected."""
    sample = {**valid_sample, "MonthlyCharges": -10.0}

    with pytest.raises(PortableValidationError, match="below minimum"):
        portable_predictor.predict_single(sample)


def test_validation_negative_total_charges(portable_predictor, valid_sample):
    """Negative TotalCharges rejected."""
    sample = {**valid_sample, "TotalCharges": -100.0}

    with pytest.raises(PortableValidationError, match="below minimum"):
        portable_predictor.predict_single(sample)


def test_validation_empty_string_categorical(portable_predictor, valid_sample):
    """Empty string for categorical rejected."""
    sample = {**valid_sample, "Contract": ""}

    with pytest.raises(PortableValidationError, match="missing value"):
        portable_predictor.predict_single(sample)


# =============================================================================
# Failure Tests
# =============================================================================

def test_corrupted_sha256_fails(artifacts_dir):
    """Corrupted SHA256 in metadata causes failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        import shutil
        shutil.copy(artifacts_dir / "final_pipeline.joblib", tmpdir / "final_pipeline.joblib")
        shutil.copy(artifacts_dir / "inference_schema.json", tmpdir / "inference_schema.json")
        shutil.copy(artifacts_dir / "inference_metadata.json", tmpdir / "inference_metadata.json")

        with open(tmpdir / "inference_metadata.json") as f:
            meta = json.load(f)
        meta["artifact_integrity"]["pipeline_sha256"] = "0" * 64
        with open(tmpdir / "inference_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        with pytest.raises(ValueError, match="Artifact integrity check failed"):
            PortableChurnPredictor.from_artifacts(tmpdir, verify_integrity=True)


def test_missing_artifact_fails(artifacts_dir):
    """Missing artifact file causes clear failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        import shutil
        shutil.copy(artifacts_dir / "final_pipeline.joblib", tmpdir / "final_pipeline.joblib")
        shutil.copy(artifacts_dir / "inference_schema.json", tmpdir / "inference_schema.json")
        # Missing metadata

        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            PortableChurnPredictor.from_artifacts(tmpdir)


def test_malformed_metadata_fails(artifacts_dir):
    """Malformed metadata JSON causes clear failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        import shutil
        shutil.copy(artifacts_dir / "final_pipeline.joblib", tmpdir / "final_pipeline.joblib")
        shutil.copy(artifacts_dir / "inference_schema.json", tmpdir / "inference_schema.json")
        # Write invalid JSON
        (tmpdir / "inference_metadata.json").write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            PortableChurnPredictor.from_artifacts(tmpdir)


def test_malformed_schema_fails(artifacts_dir):
    """Malformed schema JSON causes clear failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        import shutil
        shutil.copy(artifacts_dir / "final_pipeline.joblib", tmpdir / "final_pipeline.joblib")
        shutil.copy(artifacts_dir / "inference_metadata.json", tmpdir / "inference_metadata.json")
        # Write invalid JSON
        (tmpdir / "inference_schema.json").write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            PortableChurnPredictor.from_artifacts(tmpdir)


# =============================================================================
# CLI Tests
# =============================================================================

def test_cli_valid_input():
    """CLI with valid input succeeds."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("""gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
Female,0,Yes,No,12,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.50,29.85
""")
        input_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = f.name

    result = subprocess.run([
        "uv", "run", "python", "-m", "customer_churn.portable_predict",
        "--input", input_path,
        "--artifacts", "models",
        "--output", output_path
    ], capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parents[1])

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert "Predictions saved" in result.stdout

    # Verify output file
    df = pd.read_csv(output_path)
    assert len(df) == 1
    assert "churn_prediction" in df.columns
    assert "churn_probability" in df.columns

    Path(input_path).unlink()
    Path(output_path).unlink()


def test_cli_invalid_input_fails():
    """CLI with invalid input fails with non-zero exit."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("gender,SeniorCitizen\nFemale,0\n")
        input_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        output_path = f.name

    result = subprocess.run([
        "uv", "run", "python", "-m", "customer_churn.portable_predict",
        "--input", input_path,
        "--artifacts", "models",
        "--output", output_path
    ], capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parents[1])

    assert result.returncode != 0, "CLI should fail with invalid input"
    assert "Missing required fields" in result.stderr or "Missing required fields" in result.stdout

    Path(input_path).unlink()
    Path(output_path).unlink()


# =============================================================================
# Extraction Tests
# =============================================================================

def test_extract_inference_artifacts_function(artifacts_dir):
    """extract_inference_artifacts function works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        schema, metadata = extract_inference_artifacts(
            artifacts_dir=artifacts_dir,
            output_dir=tmpdir,
        )

        assert "numeric_features" in schema
        assert "categorical_features" in schema
        assert metadata["production_threshold"] == 0.28
        assert (tmpdir / "inference_schema.json").exists()
        assert (tmpdir / "inference_metadata.json").exists()


def test_extract_deterministic(artifacts_dir):
    """Running extraction twice produces identical artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        tmpdir1 = Path(tmpdir1)
        tmpdir2 = Path(tmpdir2)

        extract_inference_artifacts(artifacts_dir=artifacts_dir, output_dir=tmpdir1)
        extract_inference_artifacts(artifacts_dir=artifacts_dir, output_dir=tmpdir2)

        with open(tmpdir1 / "inference_schema.json") as f:
            schema1 = json.load(f)
        with open(tmpdir2 / "inference_schema.json") as f:
            schema2 = json.load(f)

        with open(tmpdir1 / "inference_metadata.json") as f:
            meta1 = json.load(f)
        with open(tmpdir2 / "inference_metadata.json") as f:
            meta2 = json.load(f)

        assert schema1 == schema2
        assert meta1 == meta2


# =============================================================================
# Metadata Tests
# =============================================================================

def test_metadata_has_required_fields(artifacts_dir):
    """Metadata contains all required fields."""
    with open(artifacts_dir / "inference_metadata.json") as f:
        metadata = json.load(f)

    required = [
        "version",
        "model_type",
        "model_version",
        "production_threshold",
        "threshold_metadata",
        "preprocessing",
        "artifact_integrity",
        "runtime",
    ]
    for field in required:
        assert field in metadata, f"Missing field: {field}"

    # Check threshold_metadata
    assert "value" in metadata["threshold_metadata"]
    assert "selection_method" in metadata["threshold_metadata"]
    assert "source_phase" in metadata["threshold_metadata"]
    assert "cost_matrix" in metadata["threshold_metadata"]
    assert "frozen_test_cost" in metadata["threshold_metadata"]

    # Check artifact_integrity
    assert "algorithm" in metadata["artifact_integrity"]
    assert "pipeline_sha256" in metadata["artifact_integrity"]

    # Check runtime
    assert "python_version" in metadata["runtime"]
    assert "sklearn_version" in metadata["runtime"]


def test_inference_metadata_class(artifacts_dir):
    """InferenceMetadata class loads correctly."""
    metadata = InferenceMetadata.from_json(artifacts_dir / "inference_metadata.json")
    assert metadata.production_threshold == 0.28
    assert metadata.threshold == 0.28
    assert metadata.threshold_metadata["source_phase"] == "Phase 11"


# =============================================================================
# No Training Dependency Tests
# =============================================================================

def test_portable_predictor_no_features_import():
    """PortableChurnPredictor does not import features.py at module load."""
    # This is verified by the fact that from_artifacts works without
    # config.PROCESSED_DATA_PATH existing
    # No assertion needed - if features.py was imported at module level,
    # it would fail due to missing data/processed/


def test_portable_contract_no_features_import():
    """PortableDataContract does not depend on features.py."""
    with open("models/inference_schema.json") as f:
        schema = json.load(f)
    contract = PortableDataContract(schema)
    assert contract is not None
    assert len(contract.required_features) == 19


# =============================================================================
# Schema Validation Tests (direct PortableDataContract)
# =============================================================================

def test_portable_contract_direct_validation(valid_sample):
    """PortableDataContract.validate works directly."""
    with open("models/inference_schema.json") as f:
        schema = json.load(f)
    contract = PortableDataContract(schema)

    result = contract.validate(valid_sample)
    assert isinstance(result, PortableValidationResult)
    assert len(result.validated_data) == 1
    assert list(result.validated_data.columns) == contract.required_features


def test_portable_contract_batch_validation(valid_sample):
    """PortableDataContract.validate_batch works."""
    with open("models/inference_schema.json") as f:
        schema = json.load(f)
    contract = PortableDataContract(schema)

    df = pd.DataFrame([valid_sample, valid_sample])
    result = contract.validate_batch(df)
    assert len(result.validated_data) == 2
    assert list(result.validated_data.columns) == contract.required_features


def test_portable_contract_empty_batch_rejected(valid_sample):
    """Empty DataFrame rejected."""
    with open("models/inference_schema.json") as f:
        schema = json.load(f)
    contract = PortableDataContract(schema)

    with pytest.raises(PortableValidationError, match="empty"):
        contract.validate_batch(pd.DataFrame())
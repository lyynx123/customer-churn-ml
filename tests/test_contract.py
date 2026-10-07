"""Tests for Phase 14 - Data Contract."""

import pytest

from customer_churn.contract import (
    DataContract,
    ValidationError,
    ValidationResult,
    derive_schema_from_pipeline,
)
from customer_churn.features import NUMERIC_COLS, _get_categorical_columns
from customer_churn.predict import (
    DEFAULT_THRESHOLD,
    THRESHOLD_METADATA,
    ChurnPredictor,
    load_pipeline,
)

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def production_pipeline():
    """Load the production pipeline for testing."""
    return load_pipeline()


@pytest.fixture(scope="module")
def production_preprocessor():
    """Get the production preprocessor."""
    pipeline = load_pipeline()
    return pipeline.named_steps["preprocessor"]


@pytest.fixture(scope="module")
def production_contract(production_pipeline):
    """Create a DataContract with the production pipeline."""
    return DataContract(production_pipeline)


@pytest.fixture(scope="module")
def contract(production_contract):
    """Alias for production_contract for backward compatibility."""
    return production_contract


@pytest.fixture(scope="module")
def valid_single_input():
    """A valid single-row input for testing."""
    return {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }


# ============================================================
# 1. Schema Consistency with Production Pipeline
# ============================================================

def test_schema_consistency_with_production_pipeline(contract):
    """Contract schema must exactly match the fitted production pipeline."""

    # Numeric features match
    assert set(contract.numeric_features) == set(NUMERIC_COLS)

    # Categorical features match (order and names)
    assert list(contract.categorical_features.keys()) == _get_categorical_columns()

    # Allowed values match fitted encoder exactly
    pipeline = load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    for i, col in enumerate(_get_categorical_columns()):
        allowed = contract.categorical_features[col]["allowed"]
        encoder_cats = list(cat_encoder.categories_[i])
        assert set(allowed) == set(encoder_cats), f"Mismatch for {col}: {allowed} vs {encoder_cats}"


def test_derive_schema_from_pipeline_matches_contract():
    """derive_schema_from_pipeline should produce schema consistent with DataContract."""
    pipeline = load_pipeline()
    schema = derive_schema_from_pipeline(pipeline)

    assert set(schema["numeric_features"]) == set(NUMERIC_COLS)
    assert list(schema["categorical_features"].keys()) == _get_categorical_columns()
    assert set(schema["forbidden_fields"]) == {"customerID", "Churn"}
    assert schema["feature_order"] == list(NUMERIC_COLS) + _get_categorical_columns()


# ============================================================
# 2. Valid Input
# ============================================================

def test_valid_input_passes(contract, valid_single_input):
    """Complete valid input passes validation."""
    import pandas as pd
    result = contract.validate(valid_single_input)

    assert isinstance(result, ValidationResult)
    assert isinstance(result.validated_data, pd.DataFrame)
    assert len(result.validated_data) == 1
    assert list(result.validated_data.columns) == contract.required_features


def test_valid_batch_passes(contract, valid_single_input):
    """Valid batch input passes validation."""
    import pandas as pd
    result = contract.validate_batch(pd.DataFrame([valid_single_input, valid_single_input]))

    assert isinstance(result, ValidationResult)
    assert len(result.validated_data) == 2
    assert list(result.validated_data.columns) == contract.required_features


# ============================================================
# 3. Missing Required Field
# ============================================================

def test_missing_required_field_fails(contract):
    """Missing any required field fails validation."""
    # Remove a numeric field
    input_data = {
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Missing required fields" in str(exc_info.value)
    assert "MonthlyCharges" in str(exc_info.value)


def test_missing_categorical_field_fails(contract):
    """Missing a categorical field fails validation."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Missing required fields" in str(exc_info.value)
    assert "Contract" in str(exc_info.value)


# ============================================================
# 4. Forbidden Fields
# ============================================================

def test_forbidden_customerID_rejected(contract):
    """customerID in input is rejected."""
    input_data = {
        "customerID": "12345",
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Forbidden fields present" in str(exc_info.value)
    assert "customerID" in str(exc_info.value)


def test_forbidden_Churn_rejected(contract):
    """Churn target in input is rejected."""
    input_data = {
        "Churn": "Yes",
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Forbidden fields present" in str(exc_info.value)
    assert "Churn" in str(exc_info.value)


# ============================================================
# 5. Unexpected Fields
# ============================================================

def test_unexpected_field_rejected(contract):
    """Unexpected/extra fields are rejected in strict mode."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
        "unexpected_field": "should_fail",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Unexpected fields" in str(exc_info.value)
    assert "unexpected_field" in str(exc_info.value)


# ============================================================
# 6. Numeric Type Validation
# ============================================================

def test_invalid_numeric_type_fails(contract):
    """String for numeric field fails."""
    input_data = {
        "MonthlyCharges": "seventy-five",  # string instead of number
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "MonthlyCharges" in str(exc_info.value)
    assert "invalid" in str(exc_info.value).lower()


# ============================================================
# 7. Non-finite Numeric Values
# ============================================================

def test_nan_numeric_fails(contract):
    """NaN in numeric field fails."""
    input_data = {
        "MonthlyCharges": float("nan"),
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "MonthlyCharges" in str(exc_info.value)
    assert "non-finite" in str(exc_info.value).lower() or "missing" in str(exc_info.value).lower()


def test_inf_numeric_fails(contract):
    """inf in numeric field fails."""
    input_data = {
        "MonthlyCharges": float("inf"),
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "MonthlyCharges" in str(exc_info.value)
    assert "non-finite" in str(exc_info.value).lower()


def test_negative_inf_numeric_fails(contract):
    """-inf in numeric field fails."""
    input_data = {
        "MonthlyCharges": float("-inf"),
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "MonthlyCharges" in str(exc_info.value)
    assert "non-finite" in str(exc_info.value).lower()


# ============================================================
# 8. Numeric Constraints
# ============================================================

def test_senior_citizen_constraint(contract):
    """SeniorCitizen must be 0 or 1."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 2,  # invalid
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "SeniorCitizen" in str(exc_info.value)
    assert "allowed" in str(exc_info.value).lower()


def test_tenure_min_constraint(contract):
    """tenure must be >= 0."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": -5,  # invalid
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "tenure" in str(exc_info.value)
    assert "minimum" in str(exc_info.value).lower()


def test_monthly_charges_min_constraint(contract):
    """MonthlyCharges must be >= 0."""
    input_data = {
        "MonthlyCharges": -10.0,  # invalid
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "MonthlyCharges" in str(exc_info.value)
    assert "minimum" in str(exc_info.value).lower()


def test_total_charges_min_constraint(contract):
    """TotalCharges must be >= 0."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": -100.0,  # invalid
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "TotalCharges" in str(exc_info.value)
    assert "minimum" in str(exc_info.value).lower()


# ============================================================
# 9. Categorical Values
# ============================================================

def test_invalid_categorical_value_fails(contract):
    """Value not in allowed categories fails."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Yearly",  # invalid - not in allowed
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Contract" in str(exc_info.value)
    assert "allowed categories" in str(exc_info.value)


def test_categorical_case_sensitivity(contract):
    """Categorical values are case-sensitive."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "month-to-month",  # lowercase - invalid
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Contract" in str(exc_info.value)


# ============================================================
# 10. Unknown Categorical Values
# ============================================================

def test_unknown_categorical_strict_fails(contract):
    """Unknown category rejected in strict (default) mode."""
    # Default contract is already strict (allow_unknown_categories=False)
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Other",  # not in allowed
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline(), allow_unknown_categories=False).validate(input_data)

    assert "gender" in str(exc_info.value)


def test_unknown_categorical_lenient_passes():
    """Unknown category passes in lenient mode (allow_unknown=True)."""
    lenient_contract = DataContract(load_pipeline(), allow_unknown_categories=True)
    _ = lenient_contract  # silence unused warning
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Other",  # not in allowed
    }

    # Should not raise in lenient mode
    result = DataContract(load_pipeline(), allow_unknown_categories=True).validate(input_data)
    assert isinstance(result, ValidationResult)


# ============================================================
# 11. Missing Values
# ============================================================

def test_missing_numeric_value_fails(contract):
    """None/NaN for numeric field fails."""
    input_data = {
        "MonthlyCharges": None,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "MonthlyCharges" in str(exc_info.value)
    assert "missing" in str(exc_info.value).lower()


def test_missing_categorical_value_fails(contract):
    """None/empty for categorical field fails."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": None,
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Contract" in str(exc_info.value)
    assert "missing" in str(exc_info.value).lower()


def test_empty_string_categorical_fails(contract):
    """Empty string for categorical field fails."""
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }

    with pytest.raises(ValidationError) as exc_info:
        DataContract(load_pipeline()).validate(input_data)

    assert "Contract" in str(exc_info.value)
    assert "missing" in str(exc_info.value).lower()


# ============================================================
# 12. Feature Ordering Consistency
# ============================================================

def test_feature_ordering_consistency(contract, valid_single_input):
    """Validated data columns must match preprocessor expected order."""
    result = contract.validate(valid_single_input)
    assert list(result.validated_data.columns) == contract.required_features


# ============================================================
# 13. Schema Consistency with Fitted Production Pipeline
# ============================================================

def test_schema_consistency_with_fitted_pipeline(contract):
    """Contract schema must exactly match fitted production pipeline."""
    pipeline = load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

    # Numeric features
    assert set(contract.numeric_features) == set(NUMERIC_COLS)

    # Categorical features order and names
    assert list(contract.categorical_features.keys()) == _get_categorical_columns()

    # Allowed values match exactly
    for i, col in enumerate(_get_categorical_columns()):
        allowed = set(contract.categorical_features[col]["allowed"])
        encoder_cats = set(cat_encoder.categories_[i])
        assert allowed == encoder_cats, f"Mismatch for {col}"

    # Feature order matches preprocessor
    expected_order = list(NUMERIC_COLS) + _get_categorical_columns()
    assert contract.required_features == expected_order


# ============================================================
# 14. Pipeline Instance Sharing
# ============================================================

def test_contract_uses_same_pipeline_instance(production_pipeline):
    """Contract must use the exact pipeline instance passed to it."""
    contract = DataContract(production_pipeline)
    assert contract._pipeline is production_pipeline


def test_churn_predictor_uses_same_pipeline():
    """ChurnPredictor and its DataContract share the same pipeline instance."""
    predictor = ChurnPredictor()
    contract = predictor._contract

    assert contract._pipeline is predictor.pipeline
    assert contract._preprocessor is predictor.pipeline.named_steps["preprocessor"]
    assert contract._model is predictor.pipeline.named_steps["model"]


def test_custom_pipeline_uses_passed_instance():
    """DataContract uses the pipeline instance passed to it."""
    custom_pipeline = load_pipeline()
    contract = DataContract(custom_pipeline)
    assert contract._pipeline is custom_pipeline


# ============================================================
# 15. No Model Mutation
# ============================================================

def test_no_model_mutation_during_validation():
    """Validation must not mutate the model or preprocessor."""
    pipeline = load_pipeline()
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    original_n_estimators = model.n_estimators
    original_max_depth = model.max_depth

    contract = DataContract(pipeline)

    # Run validation
    input_data = {
        "MonthlyCharges": 75.5,
        "SeniorCitizen": 0,
        "TotalCharges": 1500.0,
        "tenure": 12,
        "Contract": "Month-to-month",
        "Dependents": "No",
        "DeviceProtection": "No",
        "InternetService": "DSL",
        "MultipleLines": "No",
        "OnlineBackup": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "Partner": "No",
        "PaymentMethod": "Electronic check",
        "PhoneService": "Yes",
        "StreamingMovies": "No",
        "StreamingTV": "No",
        "TechSupport": "No",
        "gender": "Female",
    }
    contract.validate(input_data)

    # Model should be unchanged
    assert model.n_estimators == original_n_estimators
    assert model.max_depth == original_max_depth

    # Preprocessor should not be refitted
    assert hasattr(preprocessor, "transformers_")


# ============================================================
# 16. Single Validation Per Public API
# ============================================================

def test_predict_proba_validates_once(monkeypatch):
    """predict_proba validates exactly once."""
    import pandas as pd
    predictor = ChurnPredictor()
    call_count = 0

    original_validate = predictor._contract.validate

    def counting_validate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(predictor._contract, "validate", counting_validate)

    X = pd.DataFrame([{
        "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
        "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
        "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
        "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
        "PaymentMethod": "Electronic check", "PhoneService": "Yes",
        "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
    }])

    predictor.predict_proba(X)
    assert call_count == 1, f"predict_proba called validate {call_count} times, expected 1"


def test_predict_validates_once(monkeypatch):
    """predict validates exactly once."""
    import pandas as pd
    predictor = ChurnPredictor()
    call_count = 0

    original_validate = predictor._contract.validate

    def counting_validate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(predictor._contract, "validate", counting_validate)

    X = pd.DataFrame([{
        "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
        "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
        "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
        "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
        "PaymentMethod": "Electronic check", "PhoneService": "Yes",
        "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
    }])

    predictor.predict(X)
    assert call_count == 1, f"predict called validate {call_count} times, expected 1"


def test_predict_batch_validates_once(monkeypatch):
    """predict_batch validates exactly once."""
    import pandas as pd
    predictor = ChurnPredictor()
    call_count = 0

    original_validate_batch = predictor._contract.validate_batch

    def counting_validate_batch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_validate_batch(*args, **kwargs)

    monkeypatch.setattr(predictor._contract, "validate_batch", counting_validate_batch)

    X = pd.DataFrame([{
        "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
        "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
        "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
        "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
        "PaymentMethod": "Electronic check", "PhoneService": "Yes",
        "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
    }, {
        "MonthlyCharges": 80.0, "SeniorCitizen": 1, "TotalCharges": 2000.0, "tenure": 24,
        "Contract": "One year", "Dependents": "Yes", "DeviceProtection": "Yes",
        "InternetService": "Fiber optic", "MultipleLines": "Yes", "OnlineBackup": "Yes",
        "OnlineSecurity": "Yes", "PaperlessBilling": "No", "Partner": "Yes",
        "PaymentMethod": "Credit card (automatic)", "PhoneService": "Yes",
        "StreamingMovies": "Yes", "StreamingTV": "Yes", "TechSupport": "Yes", "gender": "Male"
    }])

    predictor.predict_batch(X)
    assert call_count == 1, f"predict_batch called validate_batch {call_count} times, expected 1"


def test_predict_single_validates_once(monkeypatch):
    """predict_single validates exactly once."""
    predictor = ChurnPredictor()
    call_count = 0

    original_validate = predictor._contract.validate

    def counting_validate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(predictor._contract, "validate", counting_validate)

    customer = {
        "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
        "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
        "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
        "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
        "PaymentMethod": "Electronic check", "PhoneService": "Yes",
        "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
    }

    predictor.predict_single(customer)
    assert call_count == 1, f"predict_single called validate {call_count} times, expected 1"


# ============================================================
# 18. Production Threshold
# ============================================================

def test_production_threshold_is_028():
    """Production threshold must be 0.28 per Phase 11."""
    assert DEFAULT_THRESHOLD == 0.28


def test_threshold_metadata_reflects_phase11():
    """Threshold metadata must reflect Phase 11 business cost optimization."""
    assert THRESHOLD_METADATA["value"] == 0.28
    assert THRESHOLD_METADATA["source_phase"] == "Phase 11"
    assert "business cost" in THRESHOLD_METADATA["selection_method"].lower()


def test_churn_predictor_default_threshold_is_028():
    """ChurnPredictor default threshold is 0.28."""
    predictor = ChurnPredictor()
    assert predictor.threshold == 0.28
    assert predictor.threshold_metadata["value"] == 0.28
    assert predictor.threshold_metadata["source_phase"] == "Phase 11"


# ============================================================
# 19. Threshold Metadata in Outputs
# ============================================================

def test_predict_single_includes_threshold_metadata():
    """predict_single output includes threshold metadata."""
    predictor = ChurnPredictor()
    customer = {
        "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
        "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
        "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
        "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
        "PaymentMethod": "Electronic check", "PhoneService": "Yes",
        "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
    }

    result = predictor.predict_single(customer)

    assert "threshold_metadata" in result
    assert result["threshold_metadata"]["value"] == 0.28
    assert result["threshold_metadata"]["source_phase"] == "Phase 11"


def test_predict_batch_includes_threshold_metadata():
    """predict_batch output includes threshold metadata."""
    import pandas as pd
    predictor = ChurnPredictor()
    X = pd.DataFrame([{
        "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
        "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
        "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
        "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
        "PaymentMethod": "Electronic check", "PhoneService": "Yes",
        "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
    }])

    result = predictor.predict_batch(X)

    assert "threshold_metadata" in result
    assert result["threshold_metadata"]["value"] == 0.28
    assert result["threshold_metadata"]["source_phase"] == "Phase 11"


# ============================================================
# 20. Custom Pipeline Uses Same Instance
# ============================================================

def test_custom_pipeline_uses_same_instance():
    """Custom pipeline passed to ChurnPredictor is shared with DataContract."""
    custom_pipeline = load_pipeline()
    predictor = ChurnPredictor(pipeline=custom_pipeline)

    assert predictor.pipeline is custom_pipeline
    assert predictor._contract._pipeline is custom_pipeline


# ============================================================
# 21. Phase 11/12/13 Regression
# ============================================================

def test_phase11_business_cost_artifact_unchanged():
    """Phase 11 business cost artifact must remain unchanged."""
    import json
    with open("models/business_cost_analysis.json") as f:
        data = json.load(f)

    assert data["selected_threshold"] == 0.28
    assert data["cost_assumptions"]["cost_fn"] == 100.0
    assert data["cost_assumptions"]["cost_fp"] == 10.0
    assert data["cost_assumptions"]["cost_tp"] == 10.0
    assert data["cost_assumptions"]["cost_tn"] == 0.0


def test_phase12_artifacts_unchanged():
    """Phase 12 artifacts must remain unchanged."""
    import json
    import os

    # These files should exist and be readable
    assert os.path.exists("models/calibration_analysis.json")
    assert os.path.exists("models/calibrated_model_metadata.json")

    with open("models/calibration_analysis.json") as f:
        data = json.load(f)
    assert data["selected_method"] == "sigmoid"
    assert data["frozen_test_used_for_calibration"] is False


def test_phase13_artifacts_unchanged():
    """Phase 13 artifacts must remain unchanged."""
    import os

    assert os.path.exists("models/shap_provenance.json")
    assert os.path.exists("models/shap_global_importance.csv")
    assert os.path.exists("models/shap_local_explanations.json")


def test_phase11_threshold_artifact_unchanged():
    """Phase 11 threshold artifact must remain 0.28."""
    import json
    with open("models/business_cost_analysis.json") as f:
        data = json.load(f)
    assert data["selected_threshold"] == 0.28


def test_phase8c_model_artifact_unchanged():
    """Phase 8C model artifact must remain unchanged."""
    import joblib
    pipeline = joblib.load("models/final_pipeline.joblib")
    model = pipeline.named_steps["model"]
    assert model.n_estimators == 500
    assert model.max_depth == 10
    assert model.min_samples_split == 2
    assert model.min_samples_leaf == 4
    assert model.max_features == "sqrt"
    assert model.class_weight == "balanced"
    assert model.random_state == 42


# ============================================================
# 22. Contract with Custom Pipeline
# ============================================================

def test_contract_with_custom_pipeline():
    """DataContract works with a custom pipeline instance."""
    custom_pipeline = load_pipeline()
    contract = DataContract(custom_pipeline)

    # Should validate valid input
    input_data = {
        "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
        "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
        "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
        "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
        "PaymentMethod": "Electronic check", "PhoneService": "Yes",
        "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
    }
    result = contract.validate(input_data)
    assert isinstance(result, ValidationResult)


# ============================================================
# 23. ValidationError Structure
# ============================================================

def test_validation_error_contains_errors():
    """ValidationError contains detailed error list."""
    contract = DataContract(load_pipeline())
    try:
        contract.validate({"MonthlyCharges": "invalid"})
    except ValidationError as e:
        assert hasattr(e, "errors")
        assert isinstance(e.errors, list)
        assert len(e.errors) > 0


# ============================================================
# 24. Batch Validation
# ============================================================

def test_batch_validation_reports_all_errors(contract):
    """Batch validation reports errors from all rows."""
    import pandas as pd
    df = pd.DataFrame([
        {
            "MonthlyCharges": "invalid", "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
            "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
            "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
            "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
            "PaymentMethod": "Electronic check", "PhoneService": "Yes",
            "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
        },
        {
            "MonthlyCharges": 80.0, "SeniorCitizen": 0, "TotalCharges": 2000.0, "tenure": 24,
            "Contract": "One year", "Dependents": "Yes", "DeviceProtection": "Yes",
            "InternetService": "Fiber optic", "MultipleLines": "Yes", "OnlineBackup": "Yes",
            "OnlineSecurity": "Yes", "PaperlessBilling": "No", "Partner": "Yes",
            "PaymentMethod": "Credit card (automatic)", "PhoneService": "Yes",
            "StreamingMovies": "Yes", "StreamingTV": "Yes", "TechSupport": "Yes", "gender": "Male"
        }
    ])

    with pytest.raises(ValidationError) as exc_info:
        contract.validate_batch(df)

    errors = str(exc_info.value)
    assert "MonthlyCharges" in errors
    assert "invalid" in errors.lower()


# ============================================================
# 25. Batch Validation Success
# ============================================================

def test_batch_validation_success(contract):
    """Valid batch passes validation."""
    import pandas as pd
    df = pd.DataFrame([
        {
            "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
            "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
            "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
            "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
            "PaymentMethod": "Electronic check", "PhoneService": "Yes",
            "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
        },
        {
            "MonthlyCharges": 80.0, "SeniorCitizen": 1, "TotalCharges": 2000.0, "tenure": 24,
            "Contract": "One year", "Dependents": "Yes", "DeviceProtection": "Yes",
            "InternetService": "Fiber optic", "MultipleLines": "Yes", "OnlineBackup": "Yes",
            "OnlineSecurity": "Yes", "PaperlessBilling": "No", "Partner": "Yes",
            "PaymentMethod": "Credit card (automatic)", "PhoneService": "Yes",
            "StreamingMovies": "Yes", "StreamingTV": "Yes", "TechSupport": "Yes", "gender": "Male"
        }
    ])

    result = contract.validate_batch(df)
    assert isinstance(result, ValidationResult)
    assert len(result.validated_data) == 2
    assert list(result.validated_data.columns) == contract.required_features


# ============================================================
# 26. Empty Batch Rejection
# ============================================================

def test_empty_batch_rejected(contract):
    """Empty DataFrame rejected in batch validation."""
    import pandas as pd
    df = pd.DataFrame()

    with pytest.raises(ValidationError) as exc_info:
        contract.validate_batch(df)

    assert "empty" in str(exc_info.value).lower()


# ============================================================
# 27. Contract Does Not Fit
# ============================================================

def test_contract_does_not_call_fit():
    """DataContract never calls fit on pipeline or preprocessor."""
    import joblib
    pipeline = joblib.load("models/final_pipeline.joblib")
    contract = DataContract(pipeline)

    # Verify no fit method is called by checking that the pipeline state is unchanged
    model = pipeline.named_steps["model"]
    original_state = model.__getstate__()

    # Run validation
    contract.validate({
        "MonthlyCharges": 75.5, "SeniorCitizen": 0, "TotalCharges": 1500.0, "tenure": 12,
        "Contract": "Month-to-month", "Dependents": "No", "DeviceProtection": "No",
        "InternetService": "DSL", "MultipleLines": "No", "OnlineBackup": "No",
        "OnlineSecurity": "No", "PaperlessBilling": "Yes", "Partner": "No",
        "PaymentMethod": "Electronic check", "PhoneService": "Yes",
        "StreamingMovies": "No", "StreamingTV": "No", "TechSupport": "No", "gender": "Female"
    })

    # Model state should be unchanged
    assert model.__getstate__() == original_state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
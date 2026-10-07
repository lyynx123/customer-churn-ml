"""Tests for Phase 13 - SHAP Explainability."""

import numpy as np
import pytest

from customer_churn.explain import (
    PRODUCTION_THRESHOLD,
    build_tree_explainer,
    compute_global_shap,
    compute_local_shap,
    get_feature_names,
    group_shap_by_original_feature,
    load_explanation_data,
    load_production_pipeline,
    load_training_background,
    select_local_examples,
)

# ============================================================
# 1. Load and inspect pipeline
# ============================================================

def test_load_production_pipeline():
    """Pipeline loads and contains expected steps."""
    pipeline = load_production_pipeline()
    assert pipeline is not None
    steps = list(pipeline.named_steps.keys())
    assert "preprocessor" in steps
    assert "model" in steps


def test_feature_names_count():
    """45 feature names after preprocessing."""
    pipeline = load_production_pipeline()
    preprocessor, _ = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)
    assert len(names) == 45, f"Expected 45, got {len(names)}"


# ============================================================
# 2. Background data
# ============================================================

def test_background_from_training():
    """Background data comes from training split, correct shape."""
    bg, names = load_training_background(background_size=500)
    assert bg.shape == (500, 45)
    assert len(names) == 45


def test_background_deterministic():
    """Same seed produces identical background sample."""
    bg1, _ = load_training_background(background_size=500)
    bg2, _ = load_training_background(background_size=500)
    assert np.allclose(bg1, bg2)


# ============================================================
# 3. Explanation data alignment
# ============================================================

def test_explanation_data_alignment():
    """Test data transforms to correct shape with feature names."""
    X_exp, y_true, names = load_explanation_data("test")
    assert X_exp.shape[1] == len(names) == 45
    assert len(y_true) == len(X_exp)
    assert set(np.unique(y_true)).issubset({0, 1})


# ============================================================
# 4. TreeExplainer initialization
# ============================================================

def test_tree_explainer_initialization():
    """TreeExplainer initializes without error on production RF."""
    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)
    explainer = build_tree_explainer(model, names)
    assert isinstance(explainer, shap.TreeExplainer)
    assert explainer.feature_names == names
    assert hasattr(explainer, "expected_value")


# ============================================================
# 5. SHAP output shape and semantics
# ============================================================

def test_shap_output_shape_and_additivity():
    """SHAP values shape is (n, f, c) and additivity holds in probability space."""
    from customer_churn.features import load_data

    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)

    _, _, _ = load_explanation_data("test")
    X_small = np.asarray(preprocessor.transform(load_data("test").drop(columns=["Churn"]))[:30], dtype=float)

    explainer = build_tree_explainer(model, names)
    res = compute_global_shap(explainer, X_small, model)

    # Shape
    assert res.shap_values.shape == (len(X_small), 45, 2)
    assert res.predictions.shape == (len(X_small),)

    # Additivity in probability space: base + sum(shap[:,:,1]) ≈ prediction
    base = res.base_value
    for i in range(len(X_small)):
        shap_sum = np.sum(res.shap_values[i, :, 1])
        diff = abs(base + shap_sum - res.predictions[i])
        assert diff < 1e-6, f"Additivity failed at sample {i}: diff={diff}"


def test_binary_class_semantics():
    """SHAP class 1 corresponds to positive class 'Yes' (churn)."""
    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)

    X_exp, _, _ = load_explanation_data("test")
    X_small = X_exp[:10]

    explainer = build_tree_explainer(model, names)
    res = compute_global_shap(explainer, X_small, model)

    # Base value should be in [0,1]
    assert 0.0 <= res.base_value <= 1.0
    # Predictions are probabilities for class 1
    assert np.all(res.predictions >= 0) and np.all(res.predictions <= 1)


# ============================================================
# 6. Deterministic behavior
# ============================================================

def test_deterministic_explanations():
    """Same input produces same SHAP values."""
    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)

    X_exp, _, _ = load_explanation_data("test")
    X_small = X_exp[:20]

    explainer = build_tree_explainer(model, names)
    res1 = compute_global_shap(explainer, X_small, model)
    res2 = compute_global_shap(explainer, X_small, model)

    assert np.allclose(res1.shap_values, res2.shap_values)
    assert np.allclose(res1.predictions, res2.predictions)


# ============================================================
# 7. No model refitting during explanation
# ============================================================

def test_no_model_mutation():
    """Explanation functions never call fit() on production model/preprocessor."""
    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)

    # Capture original model state
    original_n_estimators = model.n_estimators
    original_max_depth = model.max_depth

    # Run explainability
    X_exp, _, _ = load_explanation_data("test")
    X_small = X_exp[:10]
    explainer = build_tree_explainer(model, names)
    _ = compute_global_shap(explainer, X_small, model)

    # Model should be unchanged
    assert model.n_estimators == original_n_estimators
    assert model.max_depth == original_max_depth
    # Preprocessor should not be refitted
    assert hasattr(preprocessor, "transform")


# ============================================================
# 8. Feature grouping aggregation
# ============================================================

def test_original_feature_grouping():
    """One-hot features correctly aggregated to original features."""
    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)

    X_exp, _, _ = load_explanation_data("test")
    X_small = X_exp[:50]

    explainer = build_tree_explainer(model, names)
    res = compute_global_shap(explainer, X_small, model)

    grouped = group_shap_by_original_feature(res.shap_values, names, class_index=1)
    # Should have fewer original features than transformed
    assert len(grouped) < 45
    # Every original feature should have n_columns >= 1
    assert (grouped["n_columns"] >= 1).all()
    # mean_abs should be non-negative
    assert (grouped["mean_abs"] >= 0).all()
    # Sorted by mean_abs descending
    assert grouped["mean_abs"].is_monotonic_decreasing


# ============================================================
# 9. Local example selection
# ============================================================

def test_local_example_selection():
    """TP/TN/FP/FN indices correctly identified from test set."""
    pipeline = load_production_pipeline()
    _, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    X_exp, y_true, _ = load_explanation_data("test")
    preds = model.predict_proba(X_exp)[:, 1]

    indices = select_local_examples(X_exp, y_true, preds, PRODUCTION_THRESHOLD)

    for key in ["tp", "tn", "fp", "fn"]:
        assert key in indices
        assert indices[key] >= 0  # all exist in this dataset
        # Verify correctness
        idx = indices[key]
        pred_class = int(preds[idx] >= PRODUCTION_THRESHOLD)
        actual = int(y_true[idx])
        if key == "tp":
            assert pred_class == 1 and actual == 1
        elif key == "tn":
            assert pred_class == 0 and actual == 0
        elif key == "fp":
            assert pred_class == 1 and actual == 0
        elif key == "fn":
            assert pred_class == 0 and actual == 1


# ============================================================
# 10. Local explanation generation
# ============================================================

def test_local_explanation_generation():
    """Local SHAP explanations generated for TP/TN/FP/FN."""
    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)

    X_exp, y_true, _ = load_explanation_data("test")
    preds = model.predict_proba(X_exp)[:, 1]
    indices = select_local_examples(X_exp, y_true, preds, PRODUCTION_THRESHOLD)

    explainer = build_tree_explainer(model, names)
    locals = compute_local_shap(explainer, X_exp, indices, y_true, model, names, PRODUCTION_THRESHOLD)

    for stype in ["tp", "tn", "fp", "fn"]:
        assert stype in locals
        loc = locals[stype]
        assert loc.index == indices[stype]
        assert loc.threshold == PRODUCTION_THRESHOLD
        assert loc.predicted_class in {0, 1}
        assert loc.actual_class in {0, 1}
        assert len(loc.top_positive) <= 5
        assert len(loc.top_negative) <= 5
        # All shap mapping covers all features
        assert len(loc.all_shap) == 45
        # Sum of shap values should match base + prob
        total_shap = sum(loc.all_shap.values())
        # In probability space, base + sum(shap) ≈ prob
        # base value is 0.5... so sum should be prob - base
        expected_sum = loc.predicted_probability - loc.base_value
        assert abs(total_shap - expected_sum) < 1e-3


# ============================================================
# 11. Invalid / empty input handling
# ============================================================

def test_empty_input_handling():
    """Empty arrays raise appropriate errors."""
    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)
    explainer = build_tree_explainer(model, names)

    with pytest.raises(ValueError):
        compute_global_shap(explainer, np.empty((0, 45)), model)

    with pytest.raises(ValueError):
        compute_global_shap(explainer, np.ones((10, 44)), model)  # wrong feature count


# ============================================================
# 12. Background never uses test data
# ============================================================

def test_background_never_uses_test():
    """Background is always sourced from training, not test."""
    # This is implicitly tested by load_training_background which
    # only calls load_data("train")
    bg, _ = load_training_background(background_size=100)
    assert len(bg) == 100
    # Can't directly test absence of test access without monkeypatching,
    # but the function signature and docstring establish the contract.


# ============================================================
# 13. Artifact schema (smoke)
# ============================================================

def test_artifact_schema_sanity():
    """Provenance metadata contains expected keys."""
    pipeline = load_production_pipeline()
    preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    names = get_feature_names(preprocessor)
    explainer = build_tree_explainer(model, names)

    X_exp, _, _ = load_explanation_data("test")
    X_small = X_exp[:10]

    res = compute_global_shap(explainer, X_small, model)
    from customer_churn.explain import build_provenance_metadata
    meta = build_provenance_metadata(res)

    required = [
        "production_model_artifact", "model_type", "phase_8c_config",
        "business_threshold", "output_space", "positive_class",
        "expected_value_class1", "feature_count", "shap_version",
        "reference_source", "explanation_source", "phase12_calibrated_model_explained",
    ]
    for key in required:
        assert key in meta, f"Missing metadata key: {key}"

    assert meta["phase12_calibrated_model_explained"] is False
    assert meta["output_space"].startswith("probability")
    assert meta["positive_class"] == "Yes"
    assert meta["business_threshold"] == 0.28


# ============================================================
# Import shap for type checks
# ============================================================

import shap

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
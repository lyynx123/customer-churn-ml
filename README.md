# Customer Churn Prediction & Retention System

## Overview

This project implements a production-oriented machine learning pipeline to predict customer churn for a telecommunications provider. The system identifies customers at high risk of churning so that retention teams can proactively intervene and reduce revenue loss.

## Business Problem

Telecommunications providers lose significant revenue when customers churn without warning. Retention teams lack systematic methods to identify at-risk customers early enough to intervene. This project builds a binary classifier that outputs a churn probability score for each customer, enabling prioritized outreach based on risk level.

## Machine Learning Objective

**Task**: Binary classification
**Target**: `Churn` (Yes/No)
**Objective**: Predict the probability that a customer is classified as churned (`Churn=Yes`)
**Evaluation**: ROC-AUC (primary), PR-AUC, F1, Precision, Recall, Accuracy, FPR, FNR

## Dataset

**Source**: Telco Customer Churn dataset (public, IBM/Kaggle)
**Original rows**: 7,043
**Features after cleaning**: 19 input features + 1 target (after dropping `customerID`)
**Missing data**: `TotalCharges` had 11 missing values (empty strings), all for customers with `tenure=0`. In `data.py`, these are explicitly set to `0` (no billing history) before numeric coercion; no median imputation is applied at the cleaning stage.

**Target distribution** (after cleaning):
- No (stay): 5,174 (73.5%)
- Yes (churn): 1,869 (26.5%)

**Data split** (stratified, `random_state=42`):
- Training: 3,451 (49%)
- Validation: 1,479 (21%)
- Test: 2,113 (30%)

## Project Structure

```
customer_churn/
├── data/
│   ├── raw/                 # Raw CSV (not in version control)
│   └── processed/           # Processed parquet splits (train/val/test)
├── docs/                    # Project documentation
├── models/                  # Model artifacts (JSON configs, eval results)
├── notebooks/
│   └── figures/             # Generated plots (ROC, PR, CMs, threshold curves)
├── src/customer_churn/      # Core package
│   ├── __init__.py
│   ├── business_cost.py             # Business cost matrix definitions
│   ├── business_threshold_optimizer.py   # Cost-based threshold optimization (RF)
│   ├── business_threshold_optimizer_lr.py # Cost-based threshold optimization (LR)
│   ├── calibration.py               # Phase 12: probability calibration
│   ├── calibration_lr.py            # Phase 12: calibration for Tuned LR
│   ├── config.py                    # Paths, random seed
│   ├── contract.py                  # Phase 14: Data Contract
│   ├── data.py                      # Loading, cleaning, validation
│   ├── error_analysis.py            # Phase 10: descriptive error analysis
│   ├── evaluate.py                  # Phase 7: baseline test eval
│   ├── evaluate_tuned_lr.py         # Phase 9: Tuned LR test evaluation
│   ├── explain.py                   # Phase 13: SHAP explainability
│   ├── extended_business_cost.py    # Extended cost model with retention rate
│   ├── extended_threshold_optimizer.py # Extended threshold optimization
│   ├── features.py                  # Preprocessing pipeline (ColumnTransformer)
│   ├── final_evaluation.py          # Phase 8C/11: final test evaluation (RF)
│   ├── models.py                    # Baseline models, CV evaluation
│   ├── observability.py             # Phase 17: inference observability
│   ├── portable_predict.py          # Phase 15: portable inference
│   ├── predict.py                   # Inference interface & threshold management
│   ├── split.py                     # Train/val/test splitting
│   ├── threshold.py                 # Phase 8B: OOF threshold analysis
│   └── tune.py                      # Hyperparameter tuning (RandomizedSearchCV)
├── tests/                   # Unit & integration tests
├── pyproject.toml
└── README.md
```

## Methodology

The project follows a strict phase-gated methodology to prevent data leakage:

### 1. Data Cleaning (`data.py`)
- Load raw CSV, strip whitespace
- **Handle `TotalCharges` missing values**: All 11 missing entries are for `tenure=0` customers (no billing history). These are explicitly set to `0` before numeric coercion.
- Coerce `TotalCharges` to numeric (errors → NaN)
- Drop `customerID` (non-predictive identifier)
- Save cleaned parquet to `data/processed/`

### 2. Train/Validation/Test Split (`split.py`)
- Stratified 49/21/30 split on `Churn` (70/30 train-test, then 70/30 train-val)
- `random_state=42`, `shuffle=True`
- Splits saved as parquet to `data/processed/`
- **Validation set (1,479 rows, 21%)** is used for **model selection after hyperparameter tuning** (`tune.py::evaluate_on_validation`). After `RandomizedSearchCV` on training data, both tuned models are evaluated on the validation set using ROC-AUC, and the best model is selected. This gives the validation set a clear, documented purpose without changing split ratios or introducing leakage.

### 3. Preprocessing Pipeline (`features.py`)
- **Numeric** (`MonthlyCharges`, `SeniorCitizen`, `TotalCharges`, `tenure`): median imputation (fitted on training data; serves as a safety net — after cleaning, `TotalCharges` has no missing values in the current dataset)
- **Categorical** (15 columns): most-frequent imputation + OneHotEncoder (`handle_unknown='ignore'`, dense output)
- Encapsulated in `sklearn.compose.ColumnTransformer`
- Wrapped in `sklearn.pipeline.Pipeline` with the model
- **Fitted only on training data**; applied to test data during inference. Validation data is not used in the pipeline.

### 4. Baseline Modeling (`models.py`)
- 5-fold Stratified CV on training data
- Baselines: DummyClassifier, LogisticRegression, DecisionTree, RandomForest
- Metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1
- All models use identical preprocessing pipeline

### 5. Baseline Test Evaluation (`evaluate.py`)
- LogisticRegression and RandomForest evaluated on held-out test set
- Default threshold (0.50) used
- Provides baseline reference for comparison

### 6. Hyperparameter Tuning (`tune.py`) — Phase 8A
- `RandomizedSearchCV` on training data only
- LogisticRegression and RandomForestClassifier
- 5-fold StratifiedKFold, 30 iterations each
- Optimized for ROC-AUC, all metrics recorded
- Best RandomForest: `n_estimators=500, max_depth=10, min_samples_split=2, min_samples_leaf=4, max_features="sqrt", class_weight="balanced"`
- Best LogisticRegression: `C=1.0, solver="lbfgs", class_weight=None, max_iter=5000`
- **Validation-based model selection**: After tuning, both models are evaluated on the validation set (1,479 samples) using ROC-AUC. The model with the highest validation ROC-AUC is selected. This uses the validation set for its intended purpose (model selection) without leaking test data.

### 7. OOF Threshold Analysis (`threshold.py`) — Phase 8B
- Out-of-fold probability predictions via `cross_val_predict` on training data
- 5-fold StratifiedKFold, `random_state=42`
- Threshold grid: 0.10–0.90 (step 0.05)
- Predefined candidates identified:
  - **0.45**: Recall-oriented (Precision ≥ 0.50, max recall)
  - **0.50**: Default reference
  - **0.55**: F1-optimal on OOF
  - **0.60**: Precision-oriented (Recall ≥ 0.60)

### 8. Final Held-Out Test Evaluation (`final_evaluation.py`) — Phase 8C
- Fit final pipeline on **entire training set** (3,451 samples)
- Predict on **held-out test set** (2,113 samples)
- Validation set (1,479 samples) was used for model selection in Phase 8A (`tune.py`).
- Evaluate 4 predetermined thresholds: 0.45, 0.50, 0.55, 0.60
- No hyperparameter/threshold tuning on test data

## Model Selection

### Current Production Model: Tuned LogisticRegression (Phase 9/11/14/15)

```python
LogisticRegression(
    C=1.0,
    solver="lbfgs",
    class_weight=None,
    max_iter=5000,
    random_state=42,
)
```

- Selected after validation-based comparison (Phase 8A) — Tuned LR achieved higher validation ROC-AUC than Tuned RandomForest
- Serialized as `models/final_pipeline.joblib` with metadata in `models/model_metadata.json`
- Default threshold: **0.10** (Phase 11 business cost optimization on OOF probabilities for Tuned LR)

### Historical Model: RandomForestClassifier (Phase 8C/11)

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,  # execution parallelism only
)
```

- Formerly the primary model (Phases 8C, 11)
- Business-optimal threshold: **0.27** (Phase 11 on RF)
- SHAP explainability (Phase 13) generated at threshold **0.28** (artifact generation threshold)
- Calibration (Phase 12) evaluated but not adopted
- Retained as historical reference; not used in production

- `n_jobs=-1` is an execution setting (parallelism), not a hyperparameter affecting model quality

## Threshold Analysis

### Phase 8B OOF Analysis (Historical RF Reference)

| Threshold | Role | OOF Precision | OOF Recall | OOF F1 |
|-----------|------|---------------|------------|--------|
| 0.45      | Recall-oriented (Prec ≥ 0.50) | 0.5127 | 0.8144 | 0.6293 |
| 0.50      | Default reference | 0.5334 | 0.7664 | 0.6290 |
| 0.55      | F1-optimal (OOF) | 0.5657 | 0.7238 | 0.6351 |
| 0.60      | Precision-oriented (Rec ≥ 0.60) | 0.5921 | 0.6561 | 0.6225 |

### Phase 11 Business Cost Optimization

| Model | OOF-Optimal Threshold | Method |
|-------|----------------------|--------|
| **Tuned LogisticRegression (Production)** | **0.10** | Business cost minimization on OOF (Phase 11) |
| RandomForest (Historical) | 0.27 | Business cost minimization on OOF (Phase 11) |

**Threshold Distinctions (Critical):**
- **Production threshold (Tuned LR):** `0.10` — used by `ChurnPredictor` default, `DataContract`, portable inference
- **Historical RF business-optimal:** `0.27` — Phase 11 cost optimization on RandomForest
- **Historical SHAP generation:** `0.28` — threshold used when generating SHAP artifacts (Phase 13)

## Final Test Results (Production Model: Tuned LogisticRegression)

### Model-Level Metrics (Held-Out Test Set)

| Metric | Test Value |
|--------|------------|
| **ROC-AUC** | 0.8424 |
| **PR-AUC** | 0.6446 |

### Threshold Evaluation (Held-Out Test Set, Tuned LR)

| Threshold | Precision | Recall | F1 | Accuracy | FPR | FNR | Pos Rate | TN | FP | FN | TP |
|-----------|-----------|--------|----|----------|-----|-----|----------|----|----|----|----|
| 0.45      | 0.6031    | 0.6310 | 0.6167 | 0.7918 | 0.1501 | 0.3690 | 0.2778 | 1319 | 233 | 207 | 354 |
| 0.50      | 0.6419    | 0.5686 | 0.6030 | 0.8012 | 0.1147 | 0.4314 | 0.2352 | 1374 | 178 | 242 | 319 |
| 0.55      | 0.6853    | 0.4813 | 0.5654 | 0.8036 | 0.0799 | 0.5187 | 0.1865 | 1428 | 124 | 291 | 270 |
| 0.60      | 0.7045    | 0.3868 | 0.4994 | 0.7941 | 0.0586 | 0.6132 | 0.1458 | 1461 | 91  | 344 | 217 |

### Production Threshold Evaluation (Tuned LR @ 0.10)

Business cost optimization selected **0.10** as the production threshold. Test evaluation at this threshold (from Phase 11):

| Metric | Value |
|--------|-------|
| **Threshold** | 0.10 |
| **Precision** | 0.4094 |
| **Recall** | 0.9476 |
| **F1** | 0.5718 |
| **Accuracy** | 0.6233 |
| **Total Expected Cost (OOF)** | $26,000 |
| **Avg Cost/Customer (OOF)** | $7.53 |
| **Confusion Matrix** | TN=1283, FP=1252, FN=48, TP=868 |

### Historical RF Test Results (Phase 8C, for reference only)

| Threshold | Precision | Recall | F1 | Accuracy | FPR | FNR |
|-----------|-----------|--------|----|----------|-----|-----|
| 0.45      | 0.5049    | 0.8235 | 0.6260 | 0.7388 | 0.2919 | 0.1765 |
| 0.50      | 0.5234    | 0.7772 | 0.6255 | 0.7530 | 0.2558 | 0.2228 |
| 0.55      | 0.5526    | 0.7308 | 0.6293 | 0.7714 | 0.2139 | 0.2692 |
| 0.60      | 0.5684    | 0.6595 | 0.6106 | 0.7766 | 0.1811 | 0.3405 |

*These RF results are preserved for historical comparison. The production model is Tuned LR @ 0.10.*

### OOF vs. Test Generalization (Tuned LR)

| Metric | Phase 8B/11 (OOF) | Test | Δ |
|--------|-------------------|------|---|
| ROC-AUC | 0.8424 (tuning CV) | 0.8424 | ~0 |
| PR-AUC | 0.6446 (tuning CV) | 0.6446 | ~0 |

The small generalization gap suggests the model is not overfitting; calibration was evaluated (Phase 12) but not adopted for production.

## Error / Trade-off Analysis (Production Model @ 0.10)

| Aspect | Trade-off |
|--------|-----------|
| **High Recall (0.95)** | Misses very few churners (FN=48); prioritizes catching churners |
| **Lower Precision (0.41)** | More false alarms (FP=1,252); loyal customers flagged for retention |
| **Business Rationale** | Cost matrix assumes missed churn (FN) is 10x costlier than unnecessary retention (FP) |

**No single threshold is universally optimal**. Final choice requires business cost definitions:
- If missing a churner is costlier → prefer lower threshold (e.g., 0.10)
- If false alarms are costlier → prefer higher threshold (e.g., 0.55+)
- Balanced F1-optimal → 0.55 (on Tuned LR test)

> **The project implements business cost optimization (Phase 11).** Threshold selection for production is performed by minimizing an expected cost matrix based on stakeholder input.

## Reproducibility

Verified stochastic components use `random_state=42`:
- Data splitting (`train_test_split`)
- StratifiedKFold CV (`n_splits=5, shuffle=True, random_state=42`)
- RandomForest, LogisticRegression, DummyClassifier, DecisionTree
- `RandomizedSearchCV` (hyperparameter search)
- OOF `cross_val_predict`
- SHAP reference sampling

A fixed random seed improves reproducibility but does not guarantee bit-for-bit identical results in every environment. `n_jobs=-1` is used only for execution parallelism; it does not affect model determinism.

## How to Run

```bash
# 1. Install dependencies (with dev tools)
uv sync --extra dev

# 2. Run tests
uv run pytest

# 3. Lint & type-check
uv run ruff check .
uv run mypy src
uv run pyright src

# 4. Data preparation
uv run python -m customer_churn.data

# 5. Train/validation/test splitting
uv run python -m customer_churn.split

# 6. Baseline test evaluation
uv run python -m customer_churn.evaluate

# 7. Tuned LR test evaluation
uv run python -m customer_churn.evaluate_tuned_lr

# 8. Final RF evaluation (historical)
uv run python -m customer_churn.final_evaluation

# 9. Train and serialize the inference pipeline (Tuned LR)
uv run python -m customer_churn.predict

# 10. Business cost optimization (Phase 11) - selects production threshold
uv run python -m customer_churn.business_threshold_optimizer_lr

# 11. Probability calibration analysis (Phase 12) - evaluated, not adopted
uv run python -m customer_churn.calibration

# 12. SHAP explainability (Phase 13) - historical RF model
uv run python -m customer_churn.explain

# 13. Extract & verify portable inference artifacts (Phase 15)
uv run python scripts/extract_inference_artifacts.py
uv run python scripts/verify_inference_artifacts.py --artifacts-dir models

# 14. Inference observability demo (Phase 17)
uv run python -m customer_churn.observability
```

### Entry point distinction:
- `customer_churn.data` runs the data preparation pipeline.
- `customer_churn.split` creates the configured data splits.
- `customer_churn.evaluate` runs the baseline evaluation workflow for simple LogisticRegression and RandomForest models.
- `customer_churn.evaluate_tuned_lr` runs the test evaluation for the tuned LogisticRegression model.
- `customer_churn.final_evaluation` runs the final evaluation workflow for the tuned RandomForest model across multiple threshold candidates (historical).
- `customer_churn.predict` trains and serializes the inference pipeline and writes its associated metadata. It is not, by itself, the complete evaluation workflow.
- `customer_churn.business_threshold_optimizer_lr` runs Phase 11 business cost optimization for Tuned LR (production).
- `customer_churn.calibration` runs Phase 12 probability calibration analysis.
- `customer_churn.explain` runs Phase 13 SHAP explainability for historical RF.
- `customer_churn.observability` demonstrates Phase 17 metrics collection.

Note: Running these commands in sequence does not automatically reproduce every reported result. The evaluation module does not automatically rerun threshold optimization or the separate business-cost analysis.

 **Environment**: Python 3.12+, `uv` package manager

## Inference

The project provides a Python inference interface via `ChurnPredictor`:

```python
from customer_churn.predict import ChurnPredictor, load_pipeline

# Load the trained pipeline (Tuned LogisticRegression)
pipeline = load_pipeline()

# Create predictor with default threshold (0.10, business cost optimization from Phase 11 on Tuned LR)
predictor = ChurnPredictor()

# Single prediction
customer = {
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
result = predictor.predict_single(customer)
# Returns: {'churn_probability': 0.64, 'threshold': 0.10,
# 'prediction': 1, 'prediction_label': 'Yes', ...}

# Batch prediction
import pandas as pd

test_df = pd.read_parquet("data/processed/test.parquet")
X_test = test_df.drop(columns=["Churn"])
result = predictor.predict_batch(X_test.iloc[:10])
# Returns: {'predictions': [...], 'probabilities': [...], 'threshold': 0.10, ...}

# Custom threshold
result = predictor.predict_single(customer, threshold=0.45)
# Returns prediction with custom threshold
```

 **Environment**: Python 3.12+, `uv` package manager

## Business Cost Optimization (Phase 11)

**Why a new threshold?**  The F1‑optimal threshold (0.55, from Phase 8B) maximises the harmonic mean of precision and recall, but it does not consider the actual financial impact of false positives (unnecessary retention offers) versus false negatives (missed churners).  When a missed churn is substantially more costly than an unnecessary retention, the optimal operational threshold shifts.

**Cost matrix (illustrative assumptions)**
```text
cost_tn = 0.0   # No cost for true negatives (correctly predicted non‑churn)
cost_fp = 10.0  # Cost of an unnecessary retention intervention
cost_fn = 100.0 # Cost of a lost customer (missed churn)
cost_tp = 10.0  # Cost of retention intervention given to a correctly predicted churner
```
These values are **illustrative assumptions only**. Predicted churners (both FP and TP) receive a retention intervention, thus both incur the intervention cost. These assumptions must be validated or replaced with real financial and stakeholder inputs before production use.

**Decision‑layer workflow**
1. Generate Out-of-Fold (OOF) probability predictions using training data only.
2. Build a deterministic threshold grid (0.01 to 0.99, step 0.01).
3. Calculate the expected business cost for each threshold using the cost matrix.
4. Select the threshold that **minimizes total expected cost** on OOF data.
5. Evaluate the selected threshold **once** on the frozen held-out test set for final verification.

The frozen test set is strictly reserved for evaluation and does **not** participate in threshold selection.

**Reproducible command (Tuned LR - Production)**
```bash
uv run python -m customer_churn.business_threshold_optimizer_lr
```

Running the optimization with the illustrative cost matrix (FN=100, FP=10, TP=10, TN=0) on the **Tuned LogisticRegression** model yields:

- **OOF‑optimal business threshold (Tuned LR):** `0.10`
- This is the **current production threshold** used by `ChurnPredictor`.

**Artifacts generated:**
- `models/tuned_lr_business_cost_analysis.json` – machine‑readable summary of the optimisation and test evaluation.
- `models/tuned_lr_business_cost_thresholds.csv` – per‑threshold metrics and costs.
- `notebooks/figures/business_cost_curve_lr.png` – cost vs. threshold plot (minimum highlighted).
- `models/tuned_lr_cost_sensitivity_analysis.csv` – illustrative sensitivity scenarios.

**Historical RF artifacts (preserved for reference):**
- `models/business_cost_analysis.json` – RF optimization summary.
- `models/business_cost_thresholds.csv` – RF per‑threshold metrics.
- `notebooks/figures/business_cost_curve.png` – RF cost curve.

**How to customise**
Replace the `CostMatrix` values in `src/customer_churn/business_cost.py` or pass a custom `CostMatrix` instance to `run_phase11()` if you need different assumptions.

---

## Testing & Code Quality

```bash
uv run pytest          # 385 tests collected (376 passed, 9 skipped, 0 failed)
uv run ruff check .    # Linting clean
uv run mypy src        # Type checking clean (src)
uv run pyright src     # Type checking clean (src); errors in test files (missing stubs)
```

**Test coverage**: 385 tests collected covering data loading, preprocessing, pipeline structure, model configs, threshold optimization, metric correctness, OOF leakage checks, business cost logic, Phase 14 Data Contract validation, Phase 15 portable inference, Phase 17 observability, and privacy checks.

**CI Status (Run #42563626473)**: 376 passed, 9 skipped, 0 failed in ~148 seconds. Full suite completes without timeout.

---

## Data Contract (Phase 14)

The **Data Contract** is a production input boundary that validates inference requests *before* they reach the model pipeline. It enforces a strict schema derived from the fitted production model.

### Architecture

```
Raw Input
   ↓
ChurnPredictor public API
   ↓
DataContract.validate()
   ↓
Validated DataFrame
   ↓
Existing Production Pipeline
   ↓
P(Churn=Yes)
   ↓
Production Threshold = 0.10
   ↓
Prediction
```

### Features

The production model expects exactly **19 features**:

| Type | Features |
|------|----------|
| **Numeric (4)** | `MonthlyCharges`, `SeniorCitizen`, `TotalCharges`, `tenure` |
| **Categorical (15)** | `Contract`, `Dependents`, `DeviceProtection`, `InternetService`, `MultipleLines`, `OnlineBackup`, `OnlineSecurity`, `PaperlessBilling`, `Partner`, `PaymentMethod`, `PhoneService`, `StreamingMovies`, `StreamingTV`, `TechSupport`, `gender` |

### Forbidden fields

`customerID` (identifier) and `Churn` (target) are rejected at the contract boundary.

### Validation rules

- **Required columns**: All 19 production features must be present
- **Forbidden columns**: `customerID`, `Churn` are rejected
- **Unexpected columns**: Rejected (strict mode)
- **Missing values**: NaN/empty rejected at the boundary — no downstream imputation
- **Numeric constraints**: `SeniorCitizen ∈ {0,1}`; `tenure ≥ 0`; `MonthlyCharges ≥ 0`; `TotalCharges ≥ 0`; no NaN or infinity
- **Categorical validation**: Values must match the fitted `OneHotEncoder.categories_` — unknown categories are rejected by default
- **Schema source**: Categorical domains are derived from the fitted production pipeline's `OneHotEncoder`, not hardcoded
- **Pipeline instance**: The DataContract uses the *exact same* fitted pipeline instance as `ChurnPredictor` — no second model is loaded

### Production threshold: `0.10`

The **current production model** is a **Tuned LogisticRegression** (Phase 9/14/15), with a decision threshold of **0.10** selected in Phase 11 via business cost optimization on OOF probabilities (Tuned LR). This is the default for `ChurnPredictor`.

**Historical context**: The earlier Phase 8C/11 RandomForest model used a threshold of **0.27** (Phase 11 business cost optimization on OOF probabilities for RandomForest). Phase 12 probability calibration was evaluated but **not adopted** for the RandomForest model — the frozen-test business cost at 0.27 was `$16,650` (TN=862, FP=690, FN=46, TP=515). SHAP artifacts (Phase 13) were generated at threshold **0.28** for the historical RF model.

---

## SHAP Explainability (Phase 13)

**Purpose**: Post-hoc explanation of the **historical Phase 8C RandomForest model** (not the current production Tuned LR) to understand feature contributions to churn probability.

**Method**: `shap.TreeExplainer` with `model_output="raw"` and default `feature_perturbation="tree_path_dependent"`. The explained model output is the positive-class prediction from the historical scikit-learn RandomForest (P(churn=Yes)). Under this configuration, Tree SHAP returns values in the model's native output space (probability space for this estimator). `tree_path_dependent` is the default feature-dependence/background assumption for Tree SHAP; it uses the tree structure to define the background and does not use an interventional background dataset.

**Model explained**: Only the Phase 8C/11 uncalibrated RandomForest (`models/final_pipeline.joblib` as it existed before Phase 9) is explained. The current production model is Tuned LogisticRegression (Phase 9/14/15) with threshold 0.10. Phase 12 calibrated model is NOT explained (rejected experiment).

**Threshold note**: Historical SHAP artifacts were generated using a Random Forest decision threshold of **0.28**. The business cost optimization in Phase 11 later identified **0.27** as the optimal threshold for RF. Therefore, 0.28 represents the threshold used when generating the SHAP artifacts, while 0.27 represents the business-optimal threshold for RF.

**Leakage-safe design**:
- **Reference/reproducibility sample**: 500 deterministic samples from training split only (never test). Under `tree_path_dependent`, Tree SHAP does not use an interventional background dataset; the returned sample is retained for reproducibility and reference.
- **Explanation data**: Frozen test set used **only for post-hoc visualization**; never influences training, tuning, calibration, threshold selection, or model adoption.
- No model/preprocessor refitting during explanation.

**Output semantics**:
- Positive class = `Churn=Yes` (class index 1).
- Expected value (class 1) = expected positive-class model output under the configured tree-path-dependent assumption (~0.50).
- SHAP values in probability space: `expected_value + sum(SHAP) = P(churn=Yes)`.
- Positive SHAP → pushes toward churn; negative → pushes away.

**Artifacts generated**:
| Artifact | Path | Description |
|----------|------|-------------|
| Global importance (transformed) | `models/shap_global_importance.csv` | 45 features ranked by mean \|SHAP\| |
| Global importance (original grouped) | `models/shap_grouped_importance.csv` | 19 original features aggregated (sum of \|SHAP\| avoids one-hot cancellation) |
| Feature mapping | `models/shap_feature_mapping.json` | Transformed → original feature mapping |
| Local explanations | `models/shap_local_explanations.json` | TP/TN/FP/FN waterfall data (threshold 0.28, historical RF) |
| Provenance metadata | `models/shap_provenance.json` | Model config, SHAP version, background/explanation sources, leakage note |
| Summary plot | `notebooks/figures/shap_summary.png` | Beeswarm plot (top 20 features) |
| Bar plot | `notebooks/figures/shap_bar.png` | Top 20 mean \|SHAP\| |
| Waterfall plots | `notebooks/figures/shap_waterfall_{tp,tn,fp,fn}.png` | Per-sample contributions |

**Top global features (original, grouped)**:
| Original Feature | Mean \|SHAP\| | Mean Signed SHAP |
|------------------|--------------|------------------|
| Contract | 0.125 | -0.038 |
| InternetService | 0.061 | -0.003 |
| TechSupport | 0.053 | -0.019 |
| tenure | 0.050 | -0.019 |
| OnlineSecurity | 0.050 | -0.019 |

**Local examples (historical RF, threshold 0.28)**:
| Sample | Index | P(churn) | Pred | Actual | Top positive contributors |
|--------|-------|----------|------|--------|---------------------------|
| TP | 0 | 0.775 | 1 | 1 | Contract_Month-to-month (+0.27), TechSupport_No (+0.08), ... |
| TN | 1 | 0.044 | 0 | 0 | Contract_Two_year (-0.11), tenure (-0.09), ... |
| FP | 10 | 0.280 | 1 | 0 | Contract_Month-to-month (+0.15), MonthlyCharges (+0.04), ... |
| FN | 7 | 0.273 | 0 | 1 | Contract_Two_year (-0.09), TotalCharges (-0.07), ... |

**Reproducible command**:
```bash
uv run python -m customer_churn.explain
```

**Limitations & correct interpretation**:
- SHAP explains **model behavior**, not ground-truth causality.
- Correlated features (one-hot) share importance; grouped aggregation mitigates but does not eliminate this.
- One-hot cancellation is mitigated by using mean of sum of absolute SHAP per original feature.
- Frozen-test examples are **illustrative only**; not a statistical validation.
- Phase 12 calibrated model is NOT explained (it was evaluated but not adopted).
- Do not say "SHAP proves why a customer churns" — say "SHAP explains how the trained model's features contribute to its churn probability prediction."

---

## Phase 17: Inference Observability

**Purpose**: Lightweight, dependency-free metrics collection for production inference monitoring. Provides structured observability without external systems, global state, or privacy leakage.

### Architecture

```
PortableChurnPredictor / ChurnPredictor
   ↓
MetricsCollector (in-memory, per-process)
   ↓
MetricsEvent (structured dataclass)
   ↓
JSONL export (append-only) → log aggregation / SIEM
```

### Components

| Component | Description |
|-----------|-------------|
| `MetricsCollector` | Thread-compatible in-memory collector. Tracks invocation counts, success/failure counts, prediction distributions, latency, threshold, and model artifact hash. |
| `MetricsEvent` | Immutable dataclass for a single inference call or batch. Contains timestamp, event type, counts, latency_ms, threshold, success flag, error category. |
| `metrics_context` | Context manager for timing and automatic failure recording. Success recording is delegated to predictor methods (which have access to prediction counts). |
| `ErrorCategory` | Explicit enum: `VALIDATION_ERROR`, `INFERENCE_ERROR`, `ARTIFACT_INTEGRITY_ERROR`, `ARTIFACT_LOAD_ERROR`, `UNKNOWN_ERROR`. |
| `create_collector_from_metadata` | Factory to initialize collector from `InferenceMetadata` (artifact hash + production threshold). |

### Key Properties

- **Zero dependencies**: Pure stdlib + dataclasses; no OpenTelemetry, Prometheus client, or external SDKs.
- **Privacy-safe**: Events contain only aggregate counts and metadata — **no raw input data, no customer IDs, no individual probabilities**.
- **Artifact-aware**: Collector can be initialized with pipeline SHA256 and production threshold from `InferenceMetadata`.
- **Failure-transparent**: Metrics failures never mask inference outcomes; exceptions from validation/inference propagate normally.
- **JSONL export**: `write_jsonl(path)` appends structured events for log aggregation pipelines.

### Integration

`PortableChurnPredictor.from_artifacts()` automatically attaches a `MetricsCollector` initialized from artifact metadata. Access via `predictor.metrics_collector`.

```python
from customer_churn.portable_predict import PortableChurnPredictor

predictor = PortableChurnPredictor.from_artifacts("models/")
collector = predictor.metrics_collector

# Inference automatically records metrics
result = predictor.predict_batch(X_test)

# Inspect aggregated summary
summary = collector.get_summary()
# {'invocation_count': 1, 'success_count': 1, 'total_predictions': 100,
#  'positive_predictions': 23, 'negative_predictions': 77, ...}

# Export for observability pipeline
collector.write_jsonl(Path("logs/inference_metrics.jsonl"))
```

### Test Coverage (Phase 17)

- `MetricsCollector` success/failure recording, aggregation, JSONL export
- Error categorization (validation, artifact integrity, artifact load, unknown)
- `metrics_context` timing and failure capture
- `create_collector_from_metadata` factory
- Portable predictor integration (all predict_* methods record metrics)
- Privacy: no raw inputs, no customer IDs, no individual probabilities in events
- Backward compatibility: predictions identical with/without metrics collector
- CLI unchanged: `portable_predict` module works identically

---

## Artifacts

1. **Single held-out test split** — no temporal or geographic validation
2. **Illustrative business cost matrix** — TN=0, FP=10, FN=100, TP=10. These assumptions must be validated or replaced with real financial and stakeholder inputs before production use.
3. **No temporal validation** — data not evaluated for temporal stability
4. **Calibration evaluated but not adopted** — Phase 12 evaluated sigmoid/isotonic calibration for RandomForest; sigmoid selected but uncalibrated RF retained for historical artifact. Current production uses Tuned LR (uncalibrated) at threshold 0.10.
5. **Lightweight in-memory observability only** — Phase 17 `MetricsCollector` provides structured logging; no persistent production monitoring
6. **Static features** — no temporal/behavioral feature engineering
7. **Class imbalance** (~26.5% churn) handled via `class_weight='balanced'` only (for RF; Tuned LR uses no class weight)

## Future Work

- Temporal/rolling validation
- Probability calibration adoption (Platt scaling / isotonic regression)
- Persistent production monitoring & drift detection
- Model serialization (ONNX)
- Experiment tracking (MLflow/Weights & Biases)

## License

**Project code license:** A license for this project has not yet been declared. Do not assume that the project is licensed under MIT or another specific license.

### Dataset attribution and licensing

**Source:** Telco Customer Churn dataset (public, IBM/Kaggle).
The upstream IBM repository releases its code pattern under the **Apache License 2.0**. The dataset CSV's own licensing terms have not been verified. Use neutral attribution until the dataset license is confirmed.

### Model artifact redistribution

Redistribution terms have not been established. Review applicable dataset terms, dependency licenses, and other relevant restrictions before distributing model artifacts.

---

*Generated from verified Phase 8C/9/11/14/15/17 audit. All results reproducible with `random_state=42`.*
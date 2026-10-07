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
**Features after cleaning**: 20 (after dropping `customerID`)
**Missing data**: `TotalCharges` had 11 missing values (coerced to NaN, imputed via median)

**Target distribution** (after cleaning):
- No (stay): 5,174 (73.5%)
- Yes (churn): 1,869 (26.5%)

**Data split** (stratified, `random_state=42`):
- Training: 3,451 (70%)
- Validation: 1,479 (15%)
- Test: 2,113 (15%)

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
│   ├── business_cost.py      # Business cost matrix definitions
│   ├── business_threshold_optimizer.py # Cost-based threshold optimization
│   ├── config.py            # Paths, random seed
│   ├── data.py              # Loading, cleaning, validation
│   ├── error_analysis.py    # Phase 10: descriptive error analysis
│   ├── evaluate.py          # Phase 7: baseline test eval
│   ├── features.py          # Preprocessing pipeline (ColumnTransformer)
│   ├── final_evaluation.py  # Phase 8C/11: final test evaluation
│   ├── models.py            # Baseline models, CV evaluation
│   ├── predict.py           # Inference interface & threshold management
│   └── tune.py              # Hyperparameter tuning (RandomizedSearchCV)
├── tests/                   # Unit & integration tests
├── pyproject.toml
└── README.md
```

## Methodology

The project follows a strict phase-gated methodology to prevent data leakage:

### 1. Data Cleaning (`data.py`)
- Load raw CSV, strip whitespace
- Coerce `TotalCharges` to numeric (coerce errors → NaN)
- Drop `customerID` (non-predictive identifier)
- Save cleaned parquet to `data/processed/`

### 2. Train/Validation/Test Split (`split.py`)
- Stratified 70/15/15 split on `Churn`
- `random_state=42`, `shuffle=True`
- Splits saved as parquet to `data/processed/`

### 3. Preprocessing Pipeline (`features.py`)
- **Numeric** (`MonthlyCharges`, `SeniorCitizen`, `TotalCharges`, `tenure`): median imputation
- **Categorical** (15 columns): most-frequent imputation + OneHotEncoder (`handle_unknown='ignore'`, dense output)
- Encapsulated in `sklearn.compose.ColumnTransformer`
- Wrapped in `sklearn.pipeline.Pipeline` with the model
- **Fitted only on training data**; applied identically to val/test

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
- Evaluate 4 predetermined thresholds: 0.45, 0.50, 0.55, 0.60
- No hyperparameter/threshold tuning on test data

## Model Selection

**Final Model**: `RandomForestClassifier` (tuned)

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

- `n_jobs=-1` is an execution setting (parallelism), not a hyperparameter affecting model quality

## Threshold Analysis

Phase 8B OOF analysis identified four candidate thresholds:

| Threshold | Role | OOF Precision | OOF Recall | OOF F1 |
|-----------|------|---------------|------------|--------|
| 0.45      | Recall-oriented (Prec ≥ 0.50) | 0.5127 | 0.8144 | 0.6293 |
| 0.50      | Default reference | 0.5334 | 0.7664 | 0.6290 |
| 0.55      | F1-optimal (OOF) | 0.5657 | 0.7238 | 0.6351 |
| 0.60      | Precision-oriented (Rec ≥ 0.60) | 0.5921 | 0.6561 | 0.6225 |

## Final Test Results (Phase 8C)

### Model-Level Metrics
| Metric | Test Value |
|--------|------------|
| **ROC-AUC** | 0.8390 |
| **PR-AUC** | 0.6398 |

### Threshold Evaluation (Held-Out Test Set)

| Threshold | Precision | Recall | F1 | Accuracy | FPR | FNR | Pos Rate | TN | FP | FN | TP |
|-----------|-----------|--------|----|----------|-----|-----|----------|----|----|----|----|
| 0.45      | 0.5049    | 0.8235 | 0.6260 | 0.7388 | 0.2919 | 0.1765 | 0.4330 | 1099 | 453 | 99 | 462 |
| 0.50      | 0.5234    | 0.7772 | 0.6255 | 0.7530 | 0.2558 | 0.2228 | 0.3942 | 1155 | 397 | 125 | 436 |
| 0.55      | 0.5526    | 0.7308 | 0.6293 | 0.7714 | 0.2139 | 0.2692 | 0.3512 | 1220 | 332 | 151 | 410 |
| 0.60      | 0.5684    | 0.6595 | 0.6106 | 0.7766 | 0.1811 | 0.3405 | 0.3081 | 1271 | 281 | 191 | 370 |

### OOF vs. Test Generalization

| Metric | Phase 8B (OOF) | Phase 8C (Test) | Δ |
|--------|----------------|-----------------|---|
| ROC-AUC | 0.8481 | 0.8390 | -0.009 |
| PR-AUC | 0.6678 | 0.6398 | -0.028 |

The small generalization gap suggests the model is not overfitting; calibration was not independently validated.

## Error / Trade-off Analysis

| Threshold | Trade-off Profile |
|-----------|-------------------|
| **0.45** (recall-oriented) | Highest recall (0.82), lowest FNR (0.18) — misses fewest churners; higher FPR (0.29) means more loyal customers flagged |
| **0.50** | Default reference; balanced but not optimal on any single metric |
| **0.55** | **F1-optimal** (0.6293) — best harmonic mean; balanced precision/recall |
| **0.60** | Precision-oriented — lowest FPR (0.18), higher precision (0.57); higher FNR (0.34) |

**No single threshold is universally optimal**. Final choice requires business cost definitions:
- If missing a churner is costlier → prefer 0.45 or 0.50
- If false alarms are costlier → prefer 0.60
- Balanced F1-optimal → 0.55

> **The project now implements business cost optimization (Phase 11).** Threshold selection for production is performed by minimizing an expected cost matrix based on stakeholder input.

## Reproducibility

All stochastic components use `random_state=42`:
- Data splitting (`train_test_split`)
- StratifiedKFold CV (`n_splits=5, shuffle=True, random_state=42`)
- RandomForest, LogisticRegression, DummyClassifier, DecisionTree
- `RandomizedSearchCV` (hyperparameter search)
- OOF `cross_val_predict`

`n_jobs=-1` is used only for execution parallelism; it does not affect model determinism.

## How to Run

 ```bash
 # 1. Install dependencies
 uv sync

 # 2. Run tests
 uv run pytest

 # 3. Lint & type-check
 uv run ruff check .
 uv run mypy src
 uv run pyright

 # 4. Run full pipeline (optional — regenerates artifacts)
 uv run python -m customer_churn.split      # Creates train/val/test splits
 uv run python -m customer_churn.models     # Baseline CV comparison
 uv run python -m customer_churn.tune       # Hyperparameter tuning
 uv run python -m customer_churn.threshold  # Phase 8B OOF threshold analysis
 uv run python -m customer_churn.final_evaluation  # Phase 8C final test eval
 ```

 **Environment**: Python 3.12+, `uv` package manager

 ## Inference

 The project provides a Python inference interface via `ChurnPredictor`:

 ```python
 from customer_churn.predict import ChurnPredictor, load_pipeline

 # Load the trained pipeline
 pipeline = load_pipeline()

 # Create predictor with default threshold (0.28, business cost optimization from Phase 11)
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
 result = predictor.predict_single(sample)
 # Returns: {'churn_probability': 0.77, 'threshold': 0.28,
 # 'prediction': 1, 'prediction_label': 'Yes', ...}

 # Batch prediction
 import pandas as pd

 test_df = pd.read_parquet("data/processed/test.parquet")
 X_test = test_df.drop(columns=["Churn"])
 result = predictor.predict_batch(X_test.iloc[:10])
 # Returns: {'predictions': [...], 'probabilities': [...], 'threshold': 0.28, ...}

 # Custom threshold
 result = predictor.predict_single(sample, threshold=0.45)
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

 **Reproducible command**
 ```bash
 uv run python -m customer_churn.business_threshold_optimizer
 ```
 Running the optimization with the illustrative cost matrix (FN=100, FP=10, TP=10, TN=0) yields:

 - **OOF‑optimal business threshold:** `0.28`
 - **OOF Metrics (at 0.28):**
   - `TN=1407, FP=1128, FN=69, TP=847`
   - `Precision = 0.42886, Recall = 0.92467, F1 = 0.58596, Accuracy = 0.65314`
   - `Total expected cost = $26,650`
   - `Average cost per customer = $7.72`

 - **Frozen‑test evaluation (at 0.28):**
   - `TN=862, FP=690, FN=46, TP=515`
   - `Precision = 0.4274, Recall = 0.9180, F1 = 0.5832, Accuracy = 0.6517`
   - `Total cost = $16,650`
   - `Average cost per customer = $7.88`

 **Artifacts generated**
 - `models/business_cost_analysis.json` – machine‑readable summary of the optimisation and test evaluation.
 - `models/business_cost_thresholds.csv` – per‑threshold metrics and costs.
 - `notebooks/figures/business_cost_curve.png` – cost vs. threshold plot (minimum highlighted).
 - `models/cost_sensitivity_analysis.csv` – illustrative sensitivity scenarios showing how the optimal threshold moves as the FN/FP cost ratio changes.

 **How to customise**
 Replace the `CostMatrix` values in `src/customer_churn/business_cost.py` or pass a custom `CostMatrix` instance to `run_phase11()` if you need different assumptions.

 ---

 ## Testing & Code Quality

 ```bash
 uv run pytest          # 106 tests pass
 uv run ruff check .    # Linting clean
 uv run mypy src tests  # Type checking clean
 uv run pyright         # Type checking clean (main src)
 ```

 **Test coverage**: 106 tests covering data loading, preprocessing, pipeline structure, model configs, threshold optimization, metric correctness, OOF leakage checks, business cost logic, and Phase 14 Data Contract validation.

---

## Data Contract (Phase 14)

The **Data Contract** is a production input boundary that validates inference requests *before* they reach the model pipeline. It enforces a strict schema derived from the fitted production model.

### Architecture

```
Raw Input
   \u2193
ChurnPredictor public API
   \u2193
DataContract.validate()
   \u2193
Validated DataFrame
   \u2193
Existing Production Pipeline
   \u2193
P(Churn=Yes)
   \u2193
Production Threshold = 0.28
   \u2193
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
- **Missing values**: NaN/empty rejected at the boundary \u2014 no downstream imputation
- **Numeric constraints**: `SeniorCitizen \u2208 {0,1}`; `tenure \u2265 0`; `MonthlyCharges \u2265 0`; `TotalCharges \u2265 0`; no NaN or infinity
- **Categorical validation**: Values must match the fitted `OneHotEncoder.categories_` \u2014 unknown categories are rejected by default
- **Schema source**: Categorical domains are derived from the fitted production pipeline's `OneHotEncoder`, not hardcoded
- **Pipeline instance**: The DataContract uses the *exact same* fitted pipeline instance as `ChurnPredictor` \u2014 no second model is loaded

### Production threshold: `0.28`

The production decision threshold is **0.28**, selected in Phase 11 via business cost optimization (OOF probabilities). This is the default for `ChurnPredictor`.

Phase 12 probability calibration was evaluated but **not adopted** \u2014 the frozen-test business cost at 0.28 is `$16,650` (TN=862, FP=690, FN=46, TP=515).

---

## SHAP Explainability (Phase 13)

**Purpose**: Post-hoc explanation of the production RandomForest model to understand feature contributions to churn probability.

**Method**: `shap.TreeExplainer` with `model_output="raw"` and default `feature_perturbation="tree_path_dependent"`. The explained model output is the positive-class prediction from the production scikit-learn RandomForest (P(churn=Yes)). Under this configuration, Tree SHAP returns values in the model's native output space (probability space for this estimator). `tree_path_dependent` is the default feature-dependence/background assumption for Tree SHAP; it uses the tree structure to define the background and does not use an interventional background dataset.

**Production boundary**: Only the Phase 8C/11 uncalibrated RandomForest (`models/final_pipeline.joblib`) is explained. Phase 12 calibrated model is NOT explained (rejected experiment).

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
| Local explanations | `models/shap_local_explanations.json` | TP/TN/FP/FN waterfall data (threshold 0.28) |
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

**Local examples (threshold 0.28)**:
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

## Artifacts

1. **Single held-out test split** — no temporal or geographic validation
2. **Illustrative business cost matrix** — TN=0, FP=10, FN=100, TP=10. These assumptions must be validated or replaced with real financial and stakeholder inputs before production use.
3. **No temporal validation** — data not evaluated for temporal stability
4. **No calibration analysis** — probability scores not assessed for calibration
5. **No production monitoring** — no drift detection or performance tracking in place
6. **Static features** — no temporal/behavioral feature engineering
7. **Class imbalance** (~26.5% churn) handled via `class_weight='balanced'` only

## Future Work

- Temporal/rolling validation
- Probability calibration (Platt scaling / isotonic regression)
- Model monitoring & drift detection
- Model serialization (joblib/ONNX)
- Experiment tracking (MLflow/Weights & Biases)
- Feature importance analysis & SHAP explanations

## License

MIT License (or specify project license)

---

*Generated from verified Phase 8C audit. All results reproducible with `random_state=42`.*
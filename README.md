# Customer Churn Prediction & Retention System

## Overview

This project implements a production-ready machine learning pipeline to predict customer churn for a telecommunications provider. The system identifies customers at high risk of churning so that retention teams can proactively intervene and reduce revenue loss.

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
19:     config.py            # Paths, random seed
20:     data.py              # Loading, cleaning, validation
21:     features.py          # Preprocessing pipeline (ColumnTransformer)
22:     models.py            # Baseline models, CV evaluation
22:     tune.py              # Hyperparameter tuning (RandomizedSearchCV)
23:     threshold.py         # OOF threshold analysis (Phase 8B)
24:     evaluate.py          # Phase 7: baseline test eval
25:     final_evaluation.py  # Phase 8C: final test eval
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

> **No business cost matrix is currently defined.** Threshold selection for production requires stakeholder input on relative costs of false positives vs. false negatives.

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
uv run python -m customer_churn.tune       # Phase 8A hyperparameter tuning
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

# Create predictor with default threshold (0.55, OOF F1-optimal from Phase 8B)
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
# Returns: {'churn_probability': 0.77, 'threshold': 0.55,
# 'prediction': 1, 'prediction_label': 'Yes', ...}

# Batch prediction
import pandas as pd

test_df = pd.read_parquet("data/processed/test.parquet")
X_test = test_df.drop(columns=["Churn"])
result = predictor.predict_batch(X_test.iloc[:10])
# Returns: {'predictions': [...], 'probabilities': [...], 'threshold': 0.55, ...}

# Custom threshold
result = predictor.predict_single(sample, threshold=0.45)
# Returns prediction with custom threshold
```

**Environment**: Python 3.12+, `uv` package manager

## Testing & Code Quality

```bash
uv run pytest          # 62 tests pass
uv run ruff check .    # Linting clean
uv run mypy src        # Type checking clean
uv run pyright         # Type checking clean (main src; test files have pre-existing sklearn stub issues)
```

**Test coverage**: 62 tests covering data loading, preprocessing, pipeline structure, model configs, threshold configs, metric correctness, OOF leakage checks, and evaluation protocol.

## Artifacts

```
models/
├── tuning_results.json          # Phase 8A hyperparameter search results
├── threshold_analysis.json      # Phase 8B OOF threshold analysis
├── final_test_evaluation.json   # Phase 8C final test results
notebooks/figures/
├── final_threshold_comparison_test.png
├── final_confusion_matrix_threshold_045.png
├── final_confusion_matrix_threshold_050.png
├── final_confusion_matrix_threshold_055.png
├── final_confusion_matrix_threshold_060.png
└── ... (other Phase 7/8B figures)
```

## Limitations

1. **Single held-out test split** — no temporal or geographic validation
2. **No business cost matrix** — threshold selection assumes equal misclassification costs
3. **No temporal validation** — data not evaluated for temporal stability
4. **No calibration analysis** — probability scores not assessed for calibration
5. **No production monitoring** — no drift detection or performance tracking in place
6. **Static features** — no temporal/behavioral feature engineering
7. **Class imbalance** (~26.5% churn) handled via `class_weight='balanced'` only

## Future Work

- Business-cost-based threshold optimization
- Temporal/rolling validation
- Probability calibration (Platt scaling / isotonic regression)
- Model monitoring & drift detection
- Model serialization (joblib/ONNX) & Python inference interface
- Experiment tracking (MLflow/Weights & Biases)
- Feature importance analysis & SHAP explanations

## License

MIT License (or specify project license)

---

*Generated from verified Phase 8C audit. All results reproducible with `random_state=42`.*
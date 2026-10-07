# Model Card — Customer Churn Prediction

## 1. Model Overview

**Model Purpose**: Predict the probability that a telecommunications customer is labeled as churned (`Churn = Yes`) in the dataset.

**Prediction Target**: Binary classification of customer churn (`Churn`: `Yes` / `No`)

**Positive Class**: `Yes` (customer churns)

**Model Type**: `RandomForestClassifier` (scikit-learn)

**Final Artifact**: `models/final_pipeline.joblib` (includes preprocessing + trained RandomForestClassifier)

**Production Decision Threshold**: `0.28` (selected via Phase 11 business cost optimization on OOF probabilities)

> The historical Phase 8B F1-optimal threshold `0.55` is **not** the production threshold. See Section 9 for details.

**Inference Flow**:
```
raw customer input
    ↓
input validation (schema check, target/ID exclusion)
    ↓
data contract validation (Phase 14 production boundary)
    ↓
serialized sklearn Pipeline (preprocessing + model)
    ↓
preprocessing (imputation + one-hot encoding)
    ↓
RandomForestClassifier
    ↓
churn probability
    ↓
production decision threshold (≥ 0.28 → churn)
    ↓
binary prediction (1 = churn / 0 = no churn)
```

---

## 2. Intended Use

This model is intended for:

* **Churn risk scoring**: Assigning churn probability to individual customers for prioritized retention outreach
* **Analytical churn-risk assessment**: Supporting business analysis of churn drivers and customer segmentation
* **Experimentation and portfolio demonstration**: Demonstrating end-to-end ML pipeline best practices
* **Batch or single-customer inference** using the provided `ChurnPredictor` Python interface

The model's predictions should **support** analysis and operational decision-making, not automatically determine customer treatment without appropriate business review.

---

## 3. Out-of-Scope Use

This model is **not** suitable for:

* Using the model outside the supported feature schema
* Assuming predictions are causal explanations for churn behavior
* Treating probability outputs as guaranteed churn outcomes
* Using the model as a substitute for human/business judgment
* Assuming the threshold `0.55` is financially optimal without a business cost matrix
* Deploying without appropriate monitoring/validation for the target environment
* Applying to populations materially different from the training distribution without validation

---

## 4. Training Data

**Dataset**: Telco Customer Churn (public, IBM/Kaggle)

* **Original rows**: 7,043
* **Target**: `Churn` (Yes/No)
* **Positive class**: `Yes` → encoded as `1`
* **Identifier**: `customerID` (excluded from modeling)
* **Features after cleaning**: 20 columns (after dropping `customerID`)
* **Missing data**: `TotalCharges` had 11 missing values (coerced to NaN, imputed via median)

**Target distribution** (after cleaning):
- No (stay): 5,174 (73.5%)
- Yes (churn): 1,869 (26.5%)

**Data split** (stratified, `random_state=42`):
- Training: 3,451 (70%)
- Validation: 1,479 (15%)
- Test: 2,113 (15%)

**Random seed**: `42` (used throughout)

---

## 5. Features

### Numerical Features (4)
| Feature | Description |
|---------|-------------|
| `MonthlyCharges` | Monthly bill amount |
| `SeniorCitizen` | Binary (0/1) |
| `TotalCharges` | Cumulative charges (imputed median for missing) |
| `tenure` | Months as customer |

### Categorical Features (15)
`Contract`, `Dependents`, `DeviceProtection`, `InternetService`, `MultipleLines`, `OnlineBackup`, `OnlineSecurity`, `PaperlessBilling`, `Partner`, `PaymentMethod`, `PhoneService`, `StreamingMovies`, `StreamingTV`, `TechSupport`, `gender`

### Target
`Churn` (Yes/No) → encoded as `1` / `0`

### Identifier
`customerID` — **excluded from modeling** (non-predictive identifier)

### Preprocessing Pipeline
Encapsulated in `sklearn.compose.ColumnTransformer` within the serialized `Pipeline`:

* **Numeric** (`MonthlyCharges`, `SeniorCitizen`, `TotalCharges`, `tenure`): median imputation
* **Categorical** (15 columns): most-frequent imputation + OneHotEncoder (`handle_unknown='ignore'`, dense output)
* **Fitted only on training data**; applied identically to val/test/inference
* Encapsulated in `sklearn.pipeline.Pipeline` with the model → single serialized artifact

**Critical**: Preprocessing is fitted **only on training data**; validation/test/inference data are only transformed.

---

## 6. Model

**Final Model**: `RandomForestClassifier`

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 500 |
| `max_depth` | 10 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 4 |
| `max_features` | `"sqrt"` |
| `class_weight` | `"balanced"` |
| `random_state` | 42 |
| `n_jobs` | -1 |

* `n_jobs=-1` is an execution parallelism setting, not a hyperparameter affecting model quality.
* Preprocessing is included in the serialized pipeline (`models/final_pipeline.joblib`).

---

## 7. Training and Model Selection

### Baseline Modeling (Phase 6)
5-fold Stratified CV on training data:
- Baselines: DummyClassifier, LogisticRegression, DecisionTree, RandomForest
- Metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1
- All models use identical preprocessing pipeline

### Hyperparameter Tuning (Phase 8A)
- **Method**: `RandomizedSearchCV` (30 iterations each)
- **Search spaces**:
  - LogisticRegression: `C` (0.001–100), `solver` ∈ {lbfgs, liblinear}, `class_weight` ∈ {balanced, None}, `max_iter` ∈ [2000, 5000]
  - RandomForest: `n_estimators` ∈ [100,500], `max_depth` ∈ [None,10,20,30,50], `min_samples_split` ∈ [2,5,10], `min_samples_leaf` ∈ [1,2,4], `max_features` ∈ ["sqrt","log2",None], `class_weight` ∈ ["balanced","balanced_subsample",None]
- **CV**: 5-fold StratifiedKFold, `shuffle=True`, `random_state=42`
- **Optimization metric**: ROC-AUC
- **Best RandomForest**: `n_estimators=500, max_depth=10, min_samples_split=2, min_samples_leaf=4, max_features="sqrt", class_weight="balanced"`

### Threshold Analysis (Phase 8B)
- Out-of-fold probability predictions via `cross_val_predict` on training data
- 5-fold StratifiedKFold, `shuffle=True`, `random_state=42`
- Threshold grid: 0.10–0.95 (step 0.05)
- Predefined candidates:
  - **0.45**: Recall-oriented (Precision ≥ 0.50, max recall)
  - **0.50**: Default reference
  - **0.55**: F1-optimal on OOF
  - **0.60**: Precision-oriented (Recall ≥ 0.60)

### Final Model Selection
- Final model: **Tuned RandomForest** (above config)
- Trained on **full training set** (3,451 samples)
- Production threshold: **0.28** (Phase 11 business cost optimization on OOF probabilities; historical Phase 8B F1-optimal threshold was `0.55`)
- Held-out test evaluation on 2,113 frozen samples
- No hyperparameter/threshold tuning on test data

---

## 8. Evaluation

### OOF / Validation (Phase 8B)
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8481 |
| PR-AUC | 0.6678 |

### Held-out Test (Phase 8C, 2,113 samples)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8390 |
| PR-AUC | 0.6398 |

### Threshold Evaluation (Held-out Test Set)

| Threshold | Precision | Recall | F1 | Accuracy | FPR | FNR | Pos Rate | TN | FP | FN | TP |
|-----------|-----------|--------|----|----------|-----|-----|----------|----|----|----|----|
| 0.45 | 0.5049 | 0.8235 | 0.6260 | 0.7388 | 0.2919 | 0.1765 | 0.4330 | 1099 | 453 | 99 | 462 |
| 0.50 | 0.5234 | 0.7772 | 0.6255 | 0.7530 | 0.2558 | 0.2228 | 0.3942 | 1155 | 397 | 125 | 436 |
| **0.55** | **0.5526** | **0.7308** | **0.6293** | **0.7714** | **0.2139** | **0.2692** | **0.3512** | **1220** | **332** | **151** | **410** |
| 0.60 | 0.5684 | 0.6595 | 0.6106 | 0.7766 | 0.1811 | 0.3405 | 0.3081 | 1271 | 281 | 191 | 370 |

### OOF vs. Test Generalization

| Metric | Phase 8B (OOF) | Phase 8C (Test) | Δ |
|--------|----------------|-----------------|---|
| ROC-AUC | 0.8481 | 0.8390 | -0.009 |
| PR-AUC | 0.6678 | 0.6398 | -0.028 |

The OOF-to-test metric decrease represents an observed generalization gap between out-of-fold validation estimates and the frozen held-out test evaluation. Calibration was not evaluated in this project.

---

## 9. Decision Threshold

### Evaluated Candidates (Phase 8B OOF Analysis)

| Threshold | Role | OOF Precision | OOF Recall | OOF F1 |
|-----------|------|---------------|------------|--------|
| 0.45 | Recall-oriented (Prec ≥ 0.50) | 0.5127 | 0.8144 | 0.6293 |
| 0.50 | Default reference | 0.5334 | 0.7664 | 0.6290 |
| **0.55** | **F1-optimal (OOF)** | **0.5657** | **0.7238** | **0.6351** |
| 0.60 | Precision-oriented (Rec ≥ 0.60) | 0.5921 | 0.6561 | 0.6225 |

### Production Threshold
```text
0.28
```

**Selection method**: Phase 11 business cost optimization on OOF probabilities (cost matrix TN=0, FP=10, FN=100, TP=10; frozen-test cost @ 0.28 = $16,650).

**Historical context**: The threshold `0.55` was the F1-optimal candidate in the Phase 8B out-of-fold analysis. It is **not** the production threshold. Production uses `0.28` from Phase 11 business cost optimization.

**Important**: Phase 12 probability calibration was evaluated but **not adopted**. Phase 13 SHAP explainability remains based on the production (uncalibrated) model.

> **Do not describe 0.55 as "best", "optimal for business", or "recommended for production".**
>
> Instead: "0.28 is the production threshold from Phase 11 business cost optimization. 0.55 was the historical Phase 8B F1-optimal candidate."

---

## 10. Error Analysis

> The following error analysis was conducted in **Phase 10** using the historical Phase 8B F1-optimal threshold `0.55`. Current production uses threshold `0.28` (Phase 11); error rates at the production threshold will differ.

At threshold `0.55` (test set, historical Phase 10 analysis):

| Metric | Value |
|--------|-------|
| False Positives | 332 |
| False Negatives | 151 |
| True Positives | 410 |
| True Negatives | 1220 |
| False Positive Rate (FPR) | 21.4% |
| False Negative Rate (FNR) | 26.9% |
| Positive Prediction Rate | 35.1% |

### Error Analysis (Phase 10, Historical Threshold 0.55)

**Confusion Matrix (Phase 10, threshold 0.55)**:
```
              Predicted
              No    Yes
Actual No   1220   332
Actual Yes   151    410
```

**Error Analysis Findings (Phase 10, Historical Threshold 0.55)**:
- **False Positives** (332): Non-churners incorrectly flagged — wasted retention resources
- **False Negatives** (151): Churners missed — lost revenue opportunity
- **FPR**: 21.4% | **FNR**: 26.9%
- **High-confidence FP**: 71 cases (prob ≥ 0.80, actual No)
- **High-confidence FN**: 28 cases (prob ≤ 0.20, actual Yes)
- **Borderline (0.45–0.65)**: 371 samples, 45.3% error rate

**Subgroup Observations (Observational, Not Causal)**:
| Subgroup | Pattern |
|----------|---------|
| Month-to-month contract | Higher FP rate (48.3%), lower FN rate (17.7%) |
| Fiber optic internet | Higher FP rate (44.1%), higher FN rate (15.5%) |
| Electronic check payment | Higher FP rate (47.8%) |
| False positives | Lower mean tenure (18.2 vs 35.1 months) |
| False negatives | Higher MonthlyCharges (mean 65.0 vs TP 76.1) |

> **Important**: These are observational associations, **not causal relationships**. No causal claims are supported.

---

## 11. Limitations

1. **Single held-out test split** — no temporal or geographic validation
2. **No business cost matrix** — threshold selection assumes equal misclassification costs
3. **No temporal validation** — data not evaluated for temporal stability
4. **No calibration validation** — probability scores not assessed for calibration
5. **No production monitoring** — no drift detection or performance tracking in place
6. **Static features** — no temporal/behavioral feature engineering
7. **Class imbalance** (~26.5% churn) handled via `class_weight='balanced'` only
8. **Limited interpretability** — Random Forest is less directly interpretable than linear models
9. **No external validation** — single dataset/source
10. **Post-hoc explainability (Phase 13)** — SHAP TreeExplainer provides global feature importance, grouped original-feature aggregation, and local waterfall explanations for TP/TN/FP/FN on the frozen test set. SHAP values are in probability space for class 1 (churn). See Phase 13 artifacts and README for details.

---

## 12. Interpretability

The final model is a **Random Forest ensemble** and is less directly interpretable than linear models.

* No SHAP, feature importance, or partial dependence analysis is currently implemented
* Feature importance available via `model.feature_importances_` but not packaged in the inference interface
* Future work: SHAP explanations, permutation importance, partial dependence plots

---

## 13. Fairness / Subgroup Considerations

The project contains subgroup/error-pattern observations, but these **must not automatically be interpreted as fairness measurements**.

If documented, explicitly distinguish:
* Descriptive subgroup performance → observed error rate differences
* Fairness evaluation → requires defined fairness metrics, protected attributes, and policy
* Causal interpretation → requires causal methodology, not observational correlation

**No formal fairness evaluation has been conducted**. Subgroup error rate differences are documented as observed patterns only, without causal interpretation or fairness judgment.

---

## 14. Deployment Considerations

**Current project status**:
- Python inference interface exists (`ChurnPredictor` class)
  - `predict_single` — single customer
  - `predict_batch` — batch inference with probabilities
  - `predict_proba` — probability scores
  - `predict` — binary labels
  - Configurable threshold
- Preprocessing inside serialized pipeline
- Target excluded from features
- `customerID` excluded from features

**Not implemented / Not required**:
- REST API (FastAPI/Flask) — Phase 13: NOT REQUIRED
- Docker containerization — Phase 17: NOT REQUIRED
- CD/CD deployment — Phase 18: NOT REQUIRED
- Production monitoring — Not implemented
- Model serialization (ONNX) — Future work
- HTTP API / Docker / CD — Not required per Phase 13/17/18

**No business cost matrix defined** → threshold selection requires stakeholder-defined costs.

---

## 15. Reproducibility

* Python requirement: `>=3.12`
* `uv.lock` present
* Random seed: `42` (centralized in `config.py:RANDOM_SEED`)
* Preprocessing encapsulated in sklearn Pipeline
* Random seeds controlled:
  - Data splitting (`train_test_split`, `StratifiedKFold`)
  - RandomForest, LogisticRegression, DecisionTree, DummyClassifier
  - `RandomizedSearchCV` (hyperparameter search)
  - OOF `cross_val_predict`
* `n_jobs=-1` used only for execution parallelism (does not affect determinism)

---

## 16. Model Artifact

**Location**: `models/final_pipeline.joblib`

**SHA256**:
```
b956b8deff3aab5880d77c4a3442b60e21cf9994b888c0a19c9cdb146e57c55d
```

**Contents**:
- `preprocessor`: `ColumnTransformer` (numeric imputer + categorical imputer + OneHotEncoder)
- `model`: `RandomForestClassifier` (fitted)

**Loading**:
```python
import joblib

pipeline = joblib.load("models/final_pipeline.joblib")
proba = pipeline.predict_proba(X)[:, 1]
```

---

## 17. Monitoring and Maintenance

No production monitoring currently implemented.

Future work (if deployed):
- Input/data drift detection
- Prediction distribution drift
- Performance drift (when labels available)
- Threshold/business KPI monitoring
- Retraining triggers

---

## 18. Security and Privacy

* No secrets/credentials in model artifact
* No PII in model artifact (features are customer attributes, not raw PII)
* `customerID` excluded from features
* No network calls or external dependencies during inference
* `joblib` serialization — standard sklearn format

---

## 19. Known Risks

| Risk | Description |
|------|-------------|
| False positives | Historical Phase 10 at 0.55: 332/1552 non-churners flagged (21.4% FPR). Production threshold 0.28 will have different FPR. |
| False negatives | Historical Phase 10 at 0.55: 151/561 churners missed (26.9% FNR). Production threshold 0.28 will have different FNR. |
| Population shift | Model trained on 2020-era Telco data; may not generalize to current populations |
| Threshold mismatch | Historical threshold 0.55 was F1-optimal, not business-optimal. Production threshold 0.28 is from Phase 11 business cost optimization. |
| No calibration | Probabilities not validated as calibrated estimates (Phase 12 evaluated but not adopted) |
| Temporal drift | No temporal validation; model may degrade over time |
| Schema drift | New categories in categorical features handled via `handle_unknown='ignore'` but may degrade performance |

---

## 20. Future Improvements

* Business-cost-based threshold optimization **(completed in Phase 11 — production threshold 0.28)**
* Probability calibration (Platt scaling / isotonic regression)
* Temporal/rolling validation
* Model monitoring & drift detection
* Model serialization (ONNX) & inference API
* Deployment readiness (API/Docker) if deployment target emerges
* Calibration analysis
* Temporal validation
* Model monitoring & drift detection
* Retraining strategy / CI-CD for model updates
* Advanced interpretability (SHAP, PDPs) — **implemented in Phase 13**
* Production monitoring & drift detection
* Retraining strategy / CI-CD for model updates

---

## 21. Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pyright src
```

Expected:
```
pytest: 62 passed
ruff: PASS
format: PASS
mypy: PASS
pyright: PASS
```
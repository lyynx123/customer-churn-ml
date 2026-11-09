# Model Card — Customer Churn Prediction

## 1. Model Overview

**Model Purpose**: Predict the probability that a telecommunications customer is labeled as churned (`Churn = Yes`) in the dataset.

**Prediction Target**: Binary classification of customer churn (`Churn`: `Yes` / `No`)

**Positive Class**: `Yes` (customer churns)

**Model Type**: `LogisticRegression` (scikit-learn)

**Final Artifact**: `models/final_pipeline.joblib` (includes preprocessing + trained LogisticRegression)

**Production Decision Threshold**: `0.10` (selected via Phase 11 business cost optimization on OOF probabilities for Tuned LogisticRegression)

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
LogisticRegression
    ↓
churn probability
    ↓
production decision threshold (≥ 0.10 → churn)
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
* **Missing data**: `TotalCharges` had 11 missing values (coerced to NaN). **Fixed**: 6 of these occur where `tenure=0` (new customers with no charges) — they now receive `TotalCharges=0` instead of median imputation. The remaining 5 missing values (with `tenure > 0`) are imputed via median.

**Target distribution** (after cleaning):
- No (stay): 5,174 (73.5%)
- Yes (churn): 1,869 (26.5%)

**Data split** (stratified, `random_state=42`):
- Training: 3,451 (49%)
- Validation: 1,479 (21%)
- Test: 2,113 (30%)

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

**Final Model**: `LogisticRegression`

| Parameter | Value |
|-----------|-------|
| `C` | 1.0 |
| `solver` | `"lbfgs"` |
| `class_weight` | `None` |
| `max_iter` | 5000 |
| `random_state` | 42 |

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
- **Best LogisticRegression**: `C=1.0, solver=lbfgs, class_weight=None, max_iter=5000`
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

### Model Comparison & Selection
Both tuned models evaluated on held-out test set:

| Metric | Tuned LogisticRegression | Tuned RandomForest |
|--------|-------------------------|-------------------|
| **Test ROC-AUC** | **0.8424** | 0.8388 |
| **Test PR-AUC** | **0.6446** | 0.6391 |
| **Business Cost (Test)** | **$16,510** | $16,650 |

### Final Model Selection
- Final model: **Tuned LogisticRegression** (above config)
- Trained on **full training set** (3,451 samples)
- Production threshold: **0.10** (Phase 11 business cost optimization on OOF probabilities for Tuned LR; historical Phase 8B F1-optimal threshold was `0.55`)
- Held-out test evaluation on 2,113 frozen samples
- No hyperparameter/threshold tuning on test data

**Rationale for selecting LogisticRegression over RandomForest:**
1. Superior test ROC-AUC (0.8424 vs 0.8388) and PR-AUC (0.6446 vs 0.6391)
2. Lower business cost on test set ($16,510 vs $16,650)
3. Natively well-calibrated — no post-hoc calibration needed
4. 1,900x smaller artifact (~8 KB vs 14.9 MB)
5. 12.8x faster inference latency
6. Inherently interpretable via coefficients
7. Simpler operational pipeline (fewer hyperparameters, no calibration step)

---

## 8. Evaluation

### OOF / Validation (Phase 8A Cross-Validation)
| Metric | Tuned LR (CV) | Tuned RF (CV) |
|--------|---------------|---------------|
| ROC-AUC | **0.8481 ± 0.0103** | 0.8477 ± 0.0070 |
| PR-AUC | 0.6586 ± 0.0319 | **0.6671 ± 0.0302** |
| Accuracy | **0.8027 ± 0.0138** | 0.7598 ± 0.0047 |
| Precision | **0.6489 ± 0.0334** | 0.5332 ± 0.0071 |
| Recall | 0.5612 ± 0.0339 | **0.7653 ± 0.0241** |
| F1 | 0.6014 ± 0.0286 | **0.6283 ± 0.0082** |

### Held-out Test (Phase 8C, 2,113 samples)

| Metric | Tuned LogisticRegression | Tuned RandomForest |
|--------|-------------------------|-------------------|
| **ROC-AUC** | **0.8424** | 0.8388 |
| **PR-AUC** | **0.6446** | 0.6391 |

### Threshold Evaluation (Held-out Test Set - Tuned LogisticRegression)

| Threshold | Precision | Recall | F1 | Accuracy | FPR | FNR | Pos Rate | TN | FP | FN | TP |
|-----------|-----------|--------|----|----------|-----|-----|----------|----|----|----|----|
| 0.45 | 0.6031 | 0.6310 | 0.6167 | 0.7918 | 0.1501 | 0.3690 | 0.2778 | 1319 | 233 | 207 | 354 |
| 0.50 | 0.6419 | 0.5686 | 0.6030 | 0.8012 | 0.1147 | 0.4314 | 0.2352 | 1374 | 178 | 242 | 319 |
| 0.55 | 0.6853 | 0.4813 | 0.5654 | 0.8036 | 0.0799 | 0.5187 | 0.1865 | 1428 | 124 | 291 | 270 |
| 0.60 | 0.7045 | 0.3868 | 0.4994 | 0.7941 | 0.0586 | 0.6132 | 0.1458 | 1461 | 91 | 344 | 217 |

> **Production threshold is 0.10 (Phase 11 for Tuned LR)**, not 0.55. The 0.55 threshold was the historical Phase 8B F1-optimal candidate. See Section 9 for production threshold details.

### Threshold Evaluation (Held-out Test Set - Tuned RandomForest, for comparison)

| Threshold | Precision | Recall | F1 | Accuracy | FPR | FNR | Pos Rate | TN | FP | FN | TP |
|-----------|-----------|--------|----|----------|-----|-----|----------|----|----|----|----|
| 0.45 | 0.5049 | 0.8235 | 0.6260 | 0.7388 | 0.2919 | 0.1765 | 0.4330 | 1099 | 453 | 99 | 462 |
| 0.50 | 0.5234 | 0.7772 | 0.6255 | 0.7530 | 0.2558 | 0.2228 | 0.3942 | 1155 | 397 | 125 | 436 |
| 0.55 | 0.5526 | 0.7308 | 0.6293 | 0.7714 | 0.2139 | 0.2692 | 0.3512 | 1220 | 332 | 151 | 410 |
| 0.60 | 0.5684 | 0.6595 | 0.6106 | 0.7766 | 0.1811 | 0.3405 | 0.3081 | 1271 | 281 | 191 | 370 |

### Business Cost Optimization (Phase 11)

| Metric | Tuned LR (Threshold 0.10) | Tuned RF (Threshold 0.27) |
|--------|--------------------------|--------------------------|
| **Test Total Cost** | **$16,510** | $16,650 |
| **Test Avg Cost/Customer** | **$7.81** | $7.88 |
| **Test Precision** | 0.4043 | 0.4274 |
| **Test Recall** | **0.9376** | 0.9180 |
| **Test F1** | 0.5650 | 0.5832 |

### Probability Calibration (Phase 12)

| Model | Baseline Log-Loss | Sigmoid Log-Loss | Improvement | Status |
|-------|------------------|------------------|-------------|--------|
| **Tuned LR** | 0.4152 | 0.4153 | -0.0001 (worse) | **Native — not needed** |
| **Tuned RF** | 0.4678 | **0.4138** | **-0.0540** | Applied (sigmoid) |

### OOF vs. Test Generalization

| Metric | Phase 8A (CV) | Phase 8C (Test) | Δ |
|--------|----------------|-----------------|---|
| ROC-AUC (LR) | 0.8481 | 0.8424 | -0.006 |
| ROC-AUC (RF) | 0.8477 | 0.8388 | -0.009 |
| PR-AUC (LR) | 0.6586 | 0.6446 | -0.014 |
| PR-AUC (RF) | 0.6671 | 0.6391 | -0.028 |

The OOF-to-test metric decrease represents an observed generalization gap between out-of-fold validation estimates and the frozen held-out test evaluation. Both models show small gaps, indicating no severe overfitting.

---

## 9. Decision Threshold

### Evaluated Candidates (Phase 8B OOF Analysis - Tuned RandomForest)

| Threshold | Role | OOF Precision | OOF Recall | OOF F1 |
|-----------|------|---------------|------------|--------|
| 0.45 | Recall-oriented (Prec ≥ 0.50) | 0.5127 | 0.8144 | 0.6293 |
| 0.50 | Default reference | 0.5334 | 0.7664 | 0.6290 |
| **0.55** | **F1-optimal (OOF)** | **0.5657** | **0.7238** | **0.6351** |
| 0.60 | Precision-oriented (Rec ≥ 0.60) | 0.5921 | 0.6561 | 0.6225 |

### Evaluated Candidates (Phase 8B OOF Analysis - Tuned LogisticRegression)

| Threshold | Role | OOF Precision | OOF Recall | OOF F1 |
|-----------|------|---------------|------------|--------|
| 0.45 | Recall-oriented | 0.4094 | 0.9476 | 0.5718 |
| 0.50 | Default reference | 0.5331 | 0.7653 | 0.6284 |
| **0.55** | **F1-optimal (OOF)** | 0.5635 | 0.7216 | 0.6328 |
| 0.60 | Precision-oriented | 0.5925 | 0.6572 | 0.6232 |

### Production Threshold
```text
0.10
```

**Selection method**: Phase 11 business cost optimization on OOF probabilities for Tuned LogisticRegression (cost matrix TN=0, FP=10, FN=100, TP=10; frozen-test cost @ 0.10 = $16,510).

**Historical context**: 
- The threshold `0.55` was the F1-optimal candidate in the Phase 8B out-of-fold analysis for RandomForest.
- The threshold `0.27` was the production threshold for RandomForest from Phase 11 business cost optimization.
- **Production now uses `0.10`** from Phase 11 business cost optimization for Tuned LogisticRegression.

**Important**: Phase 12 probability calibration was evaluated for both models. Tuned LR is natively well-calibrated (no improvement from sigmoid). Tuned RF benefits from sigmoid calibration but Phase 12 was not adopted for production. Phase 13 SHAP explainability was performed on the RandomForest; interpretability for LR is via coefficients.

> **Do not describe 0.55 or 0.27 as "best", "optimal for business", or "recommended for production".**
>
> Instead: "0.10 is the production threshold from Phase 11 business cost optimization for Tuned LogisticRegression. 0.55 was the historical Phase 8B F1-optimal candidate for RandomForest. 0.27 was the Phase 11 production threshold for RandomForest."

---

## 10. Error Analysis

> The following error analysis was conducted using the production threshold `0.10` for Tuned LogisticRegression on the held-out test set.

### Production Threshold 0.10 (Test Set, Tuned LogisticRegression)

| Metric | Value |
|--------|-------|
| False Positives | 775 |
| False Negatives | 35 |
| True Positives | 526 |
| True Negatives | 777 |
| False Positive Rate (FPR) | 49.9% |
| False Negative Rate (FNR) | 6.2% |
| Positive Prediction Rate | 61.6% |

### Confusion Matrix (Production Threshold 0.10, Tuned LR)
```
              Predicted
              No    Yes
Actual No    777    775
Actual Yes    35    526
```

### Comparison: Historical Thresholds (for context)

| Threshold | Model | FPR | FNR | Pos Rate |
|-----------|-------|-----|-----|----------|
| 0.10 | Tuned LR (Production) | 49.9% | 6.2% | 61.6% |
| 0.27 | Tuned RF (Historical) | 46.2% | 7.8% | 52.4% |
| 0.55 | Tuned RF (Phase 8B F1-opt) | 21.4% | 26.9% | 35.1% |
| 0.55 | Tuned LR (Phase 8B F1-opt) | 7.9% | 51.9% | 18.6% |

### Error Analysis Findings (Production Threshold 0.10, Tuned LR)

- **False Positives** (775): Non-churners flagged for retention — higher resource cost but captures most churners
- **False Negatives** (35): Churners missed — lower than RF at 0.27 (44) and 0.55 (151)
- **FPR**: 49.9% | **FNR**: 6.2% (vs RF 0.27: FPR 46.2%, FNR 7.8%)
- **Trade-off**: Lower threshold (0.10) prioritizes recall (93.8%) over precision (40.4%), aligning with business cost where FN cost is 10× FP cost

> **Important**: The high FPR at threshold 0.10 reflects the business cost assumption (FN=100, FP=10). If real business costs differ, the optimal threshold will change. Phase 11 sensitivity analysis shows threshold varies with FN/FP cost ratio.

### Subgroup Observations (Observational, Not Causal)

| Subgroup | Pattern (at threshold 0.10) |
|----------|-----------------------------|
| Month-to-month contract | Higher FP rate, lower FN rate |
| Fiber optic internet | Higher FP rate |
| Electronic check payment | Higher FP rate |
| False positives | Lower mean tenure |
| False negatives | Higher MonthlyCharges |

> **Important**: These are observational associations, **not causal relationships**. No causal claims are supported.

---

## 11. Limitations

1. **Single held-out test split** — no temporal or geographic validation
2. **Illustrative business cost matrix** — Phase 11 cost assumptions (TN=0, FP=10, FN=100, TP=10) must be validated with real financial inputs
3. **No temporal validation** — data not evaluated for temporal stability
4. **Calibration evaluated but not needed for LR** — Phase 12 evaluated sigmoid/isotonic calibration for both models; LR is natively well-calibrated; RF benefits from sigmoid but calibration not adopted for production
5. **Lightweight in-memory observability only** — Phase 15 `MetricsCollector` provides structured logging; no persistent production monitoring
6. **Static features** — no temporal/behavioral feature engineering
7. **Class imbalance** (~26.5% churn) handled via `class_weight='balanced'` (RF) and no class weight (LR native)
8. **No external validation** — single dataset/source
9. **Interpretability** — LogisticRegression provides direct coefficient interpretability; Random Forest SHAP explanations (Phase 13) are available but for the non-production model
10. **High FPR at production threshold** — 49.9% FPR reflects business cost assumption (FN=10×FP); if real costs differ, threshold should be re-optimized

---

## 12. Interpretability

The final model is **LogisticRegression** which provides direct coefficient-based interpretability.

* **Coefficient interpretability**: Each feature's coefficient indicates direction and magnitude of effect on log-odds of churn
* **Feature importance via coefficients**: Absolute coefficient values (after standardization) indicate feature importance
* **Global interpretation**: Coefficients allow understanding which features increase/decrease churn probability
* **Local interpretation**: Individual predictions can be explained by computing feature contributions (coefficient × feature value)

Phase 13 SHAP explainability was performed on the RandomForest model (non-production). For LogisticRegression, coefficient-based interpretation is preferred and more direct.

**Top positive coefficients (increase churn probability)**:
- `Contract_Month-to-month` (large positive)
- `InternetService_Fiber optic` 
- `PaymentMethod_Electronic check`
- `MonthlyCharges`
- `OnlineSecurity_No`

**Top negative coefficients (decrease churn probability)**:
- `Contract_Two year` (large negative)
- `tenure`
- `TechSupport_Yes`
- `OnlineSecurity_Yes`
- `Partner_Yes`

Future work: permutation importance, partial dependence plots for more complex interactions.

---

## 13. Fairness / Subgroup Considerations

The project contains subgroup/error-pattern observations, but these **must not automatically be interpreted as fairness measurements**.

If documented, explicitly distinguish:
* Descriptive subgroup performance → observed error rate differences
* Fairness evaluation → requires defined fairness metrics, protected attributes, and policy
* Causal interpretation → requires causal methodology, not observational correlation

**No formal fairness evaluation has been conducted**. Subgroup error rate differences are documented as observed patterns only, without causal interpretation or fairness judgment.

**Descriptive subgroup analysis available (Phase 10 error analysis, threshold 0.55)**: error rates by `Contract`, `InternetService`, `PaymentMethod`, `gender`, `SeniorCitizen`, and numeric feature summaries (`tenure`, `MonthlyCharges`) for FP/FN subgroups. These are observational associations only.

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
- **Portable inference artifacts (Phase 15)**: `models/portable_pipeline.joblib`, `models/inference_schema.json`, `models/inference_metadata.json` — self-contained runtime with CLI (`scripts/extract_inference_artifacts.py`, `scripts/verify_inference_artifacts.py`)

**Not implemented / Not required**:
- REST API (FastAPI/Flask) — NOT REQUIRED
- Docker containerization — NOT REQUIRED
- CI/CD deployment — NOT REQUIRED
- Persistent production monitoring — Not implemented (Phase 15: lightweight in-memory observability only)
- Model serialization (ONNX) — Future work
- HTTP API / Docker / CD — Not required

**Business cost matrix defined (Phase 11, illustrative)** — production threshold 0.27 from cost optimization; stakeholder validation required before production use.

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

**SHA256**: (regenerated with Tuned LogisticRegression)
```
[Run verification script to get current SHA256]
```

**Contents**:
- `preprocessor`: `ColumnTransformer` (numeric imputer + categorical imputer + OneHotEncoder)
- `model`: `LogisticRegression` (fitted, tuned Phase 8A)

**Portable Inference Artifacts (Phase 15)**:
- `models/inference_schema.json` — feature schema derived from fitted OneHotEncoder
- `models/inference_metadata.json` — production threshold (0.10), integrity hash, pipeline metadata
- `models/portable_pipeline.joblib` — generated by `scripts/extract_inference_artifacts.py` (not pre-committed)

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
| False positives (production threshold 0.10) | 775/1,552 non-churners flagged (49.9% FPR) — higher retention resource usage |
| False negatives (production threshold 0.10) | 35/561 churners missed (6.2% FNR) — lost revenue opportunity |
| Population shift | Model trained on 2020-era Telco data; may not generalize to current populations |
| Threshold mismatch | Historical thresholds: 0.55 (RF F1-optimal), 0.27 (RF business cost). Production threshold 0.10 is for Tuned LR from Phase 11 business cost optimization. |
| No calibration needed | LR is natively well-calibrated; RF would need calibration but not used |
| Temporal drift | No temporal validation; model may degrade over time |
| Schema drift | New categories in categorical features handled via `handle_unknown='ignore'` but may degrade performance |
| tenure=0 TotalCharges imputation | **Fixed**: tenure=0 rows now receive TotalCharges=0; new-customer predictions no longer affected by median imputation |
| High FPR at 0.10 threshold | 49.9% FPR reflects illustrative cost matrix (FN=10×FP). Real business costs may warrant higher threshold. |

---

## 20. Future Improvements

* Probability calibration adoption (Platt scaling / isotonic regression)
* Temporal/rolling validation
* Persistent production monitoring & drift detection
* Model serialization (ONNX) & inference API
* Deployment readiness (API/Docker) if deployment target emerges
* Retraining strategy / CI-CD for model updates

---

## 21. Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pyright src
uv run python scripts/verify_inference_artifacts.py --artifacts-dir models
```

Expected:
```
pytest: 326+ passed
ruff: PASS
format: PASS
mypy: PASS
pyright: PASS (src); errors in test files (missing stubs)
artifact verification: PASSED
```
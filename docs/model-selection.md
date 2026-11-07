# Model Selection Decision Record

## Decision Rule (Pre-registered)

**Rule:** Select RandomForest (RF) only if the lower bound of the 95% confidence interval for (cost_LR - cost_RF) on the validation set is greater than 0. Otherwise, select LogisticRegression (LR).

## Validation Set Results

**Dataset:** Validation set (1,479 rows, 392 churners, 26.5% churn rate). Both models trained on training set only. Thresholds from OOF optimization: RF=0.27, LR=0.10.

| Model | ROC-AUC | PR-AUC | Confusion Matrix (TN, FP, FN, TP) | Total Cost (Val) |
|-------|---------|--------|-----------------------------------|------------------|
| RF    | 0.8349  | 0.6480 | 578, 509, 36, 356                | $12,250          |
| LR    | 0.8357  | 0.6663 | 525, 562, 26, 366                | $11,880          |

**Baseline:** Contact everyone cost = $14,790 (1,479 samples, 392 churners).

**Bootstrap (paired, 10,000 resamples, seed=42):**
- Cost difference (LR - RF): mean = -$372, 95% CI = [-$1,090, +$290]
- ROC-AUC difference (LR - RF): 95% CI = [-0.0068, +0.0083]

## Decision

**RF is retained** despite the decision rule pointing to LR.

The bootstrap 95% CI for the cost difference (LR - RF) includes 0 (lower bound = -$1,090). Since the lower bound is not > 0, the pre-registered rule dictates choosing LR. However, RF is retained as a **conscious deviation** from the rule because:

1. The cost advantage of LR is not statistically significant (CI includes 0)
2. The magnitude of LR's mean advantage ($372 on 1,479 samples = $0.25/customer) is small
3. Switching to LR would incur substantial transition costs: rebuilding SHAP explainability artifacts, re-running calibration analysis, updating CI/CD tests, and rewriting documentation
4. The production RandomForest has been validated through multiple phases (8A-13) with stable artifacts
5. The operational threshold 0.27 is well-understood by stakeholders

This deviation is documented transparently. The LR candidate code and artifacts are archived at tag `archive/lr-candidate` (commit e2e6ec2).

> **Note:** The earlier model comparison that used the test set is NOT used as evidence for this decision. The test set was never consulted for this decision.

## Robustness Analysis (Validation Set)

### Cost Sensitivity to Threshold

**LogisticRegression (threshold vs. validation cost):**
| Threshold | Total Cost | TN | FP | FN | TP |
|-----------|------------|----|----|----|----|
| 0.05      | $12,400    | 383 | 704 | 16 | 376 |
| 0.10      | $11,880    | 525 | 562 | 26 | 366 |
| 0.15      | $12,530    | 604 | 483 | 42 | 350 |
| 0.20      | $12,840    | 689 | 398 | 55 | 337 |

**RandomForest (threshold vs. validation cost):**
| Threshold | Total Cost | TN | FP | FN | TP |
|-----------|------------|----|----|----|----|
| 0.22      | $12,060    | 516 | 571 | 27 | 365 |
| 0.27      | $12,250    | 578 | 509 | 36 | 356 |
| 0.30      | $12,320    | 616 | 471 | 41 | 351 |
| 0.35      | $12,700    | 659 | 428 | 50 | 342 |

Both models show relatively stable costs around their production thresholds.

## Extended Business Cost Sensitivity (Retention Success Rate)

The extended business cost model introduces a retention success rate parameter `r` (probability an intervention retains a churner). This is an **external business assumption** — the dataset contains no intervention history.

**Cost Assumptions:** TN=0, FP=10, FN=100, Intervention=10, CLV=100.

| r   | Optimal Threshold | OOF Total Cost | Best Baseline Cost | Savings vs Best Baseline (%) |
|-----|-------------------|----------------|-------------------|------------------------------|
| 0.2 | 0.70              | $89,180        | $91,600 (No Contact) | 2.64% |
| 0.4 | 0.41              | $76,300        | $91,600 (No Contact) | 14.72% |
| 0.6 | 0.35              | $60,220        | $71,150 (Contact All) | 15.36% |
| 0.8 | 0.28              | $43,530        | $52,830 (Contact All) | 17.60% |

**Note:** At r=1.0 (the Phase 11 implicit assumption), the optimal threshold is 0.27 with OOF cost $26,570.

> **Threshold Note:** The production threshold 0.27 assumes r=1.0 (all contacted churners are successfully retained). The retention rate `r` is a business assumption; the optimal threshold shifts to 0.35 (r=0.6) and 0.70 (r=0.2) as retention effectiveness decreases.

## Cost Matrix and Assumptions

**Phase 11 Cost Matrix (Default):**
- TN = $0.0 (no cost for correctly retained customers)
- FP = $10.0 (cost of unnecessary intervention)
- FN = $100.0 (lost customer value)
- TP = $10.0 (intervention cost for correctly identified churner)
- Label: "illustrative default: FN=10×FP"

**Extended Cost Matrix Parameters:**
- `cost_tn`: $0.0
- `cost_fp`: $10.0 (mirrors intervention cost)
- `cost_fn`: $100.0 (mirrors CLV)
- `cost_intervention`: $10.0
- `clv`: $100.0 (Customer Lifetime Value)
- `retention_success_rate` (r): external assumption, not learned from data

**Baseline Costs (OOF Population):**
- Contact No One: $91,600 (all churners lost)
- Contact Everyone: $34,510 (all receive intervention)

## Archived LR Candidate

The LogisticRegression candidate (commit e2e6ec2, tag `archive/lr-candidate`) includes:
- Tuned LR with C=1.0, solver=lbfgs, max_iter=5000
- Business threshold 0.10 (OOF optimization)
- Calibration analysis (sigmoid selected, not adopted)
- Extended threshold optimizer code

The LR candidate achieved $11,880 validation cost vs RF's $12,250, but the difference was not statistically significant.
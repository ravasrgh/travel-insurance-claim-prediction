# Precision Risk Mitigation: Optimizing Financial Stability Through Predictive Analytics for Travel Insurance Claims

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9ACD32?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Analysis-blueviolet?style=for-the-badge)

---

## Table of Contents

- [Project Introduction](#project-introduction)
- [Business Problem Statement](#business-problem-statement)
- [Data Understanding](#data-understanding)
- [Analytical Workflow](#analytical-workflow--the-resampling-battle)
- [Evaluation Metrics & Optimization](#evaluation-metrics--optimization)
- [Key Insights (SHAP Analysis)](#key-insights-shap-analysis)
- [Business Recommendations](#business-recommendations)
- [How to Use](#how-to-use)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)

---

## Project Introduction

**Global-Guard Travel Assurance** is a travel insurance company operating in the Asia-Pacific market. Post-pandemic, the company experienced a **surge in policy sales** as millions of travelers returned to international travel with heightened risk awareness. While premium revenue increased significantly, **profit margins eroded** over the last two quarters.

An internal audit by the Finance Team revealed the root cause: **unpredictable insurance claims**. The current rule-based approach to estimating claim reserves suffers from two critical weaknesses:

| Problem | Consequence |
|---|---|
| **Over-estimation** of claim reserves | Idle capital that could be invested elsewhere: opportunity cost |
| **Under-estimation** of claim reserves | Unexpected claims force emergency fund liquidation: insolvency risk |

As the Risk Management Department, we were tasked by the Chief Risk Officer (CRO) to build a machine learning predictive model that can forecast whether a policyholder will file a claim. This model serves as the foundation for an early warning system, enabling smarter, more proactive reserve allocation and protecting the company's financial stability.

*"In the insurance industry, you don't go bankrupt from the claims you see coming. You go bankrupt from the ones you don't."*

---

## Business Problem Statement

### The Core Question

**How can we predict whether a travel insurance policyholder will file a claim, so that the company can optimally allocate claim reserves and reduce the cash flow uncertainty that threatens financial stability?**

This is a **binary classification** problem:
- **Claim = Yes (1):** Policyholder is predicted to file a claim
- **Claim = No (0):** Policyholder is predicted NOT to file a claim

### The Cost of Errors - Why FN is Worse Than FP

In insurance claim prediction, prediction errors have asymmetric financial consequences:

| Error Type | Scenario | Business Impact | Risk Level |
|---|---|---|---|
| **True Positive (TP)** | Predict: Claim, Actual: Claim | Reserve ready, claim paid smoothly | Safe |
| **True Negative (TN)** | Predict: No Claim, Actual: No Claim | Capital not locked unnecessarily | Safe |
| **False Positive (FP)** | Predict: Claim, Actual: No Claim | Over-provisioning: capital locked, possible overpriced premiums | Moderate |
| **False Negative (FN)** | Predict: No Claim, Actual: Claim | Unexpected claim: underfunded reserves, **insolvency risk**, regulatory penalties | Critical |

#### Why False Negatives Are Fatal

**False Negative = "The claim you never saw coming."** The company doesn't prepare funds because the model said "No Claim," but the policyholder files one anyway.

1. **Insolvency Risk**: If multiple FNs occur simultaneously, reserves dry up. Regulators may revoke the operating license.
2. **Reputation Damage**: Delayed claim payouts erode trust with customers and partner agencies.
3. **Regulatory Fines**: Many jurisdictions impose penalties for late claim settlements.

**Bottom line:** We'd rather be *overly cautious* (predict claim when there isn't one) than *caught off guard* (miss a real claim). Minimizing False Negatives is the #1 priority.

### Cost Assumptions

```
Per-case cost assumptions used in ROI simulation:

  False Negative (FN) = $500   Emergency investigation + unplanned reserve withdrawal
  False Positive (FP) = $50    Opportunity cost of unnecessarily locked capital
  True Positive  (TP) = $20    Routine investigation (claim was anticipated)
  True Negative  (TN) = $0     No additional cost
```

---

## Data Understanding

### Dataset Overview

| Property | Value |
|---|---|
| **Source** | Internal Global-Guard policy database |
| **Records** | 44,328 policies |
| **Features** | 10 features + 1 target |
| **Target** | `Claim` (Yes / No) |
| **Claim Rate** | ~1.53% Yes vs ~98.47% No |
| **Imbalance Ratio** | 1 : 64 (extremely imbalanced) |

### Feature Descriptions

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `Agency` | Categorical | Agent code that sold the policy |
| 2 | `Agency Type` | Categorical | Airlines or Travel Agency |
| 3 | `Distribution Channel` | Categorical | Online or Offline |
| 4 | `Product Name` | Categorical | Travel insurance product name |
| 5 | `Gender` | Categorical | Policyholder gender (F/M) (*dropped due to >30% missing*) |
| 6 | `Duration` | Numerical | Trip duration (days) |
| 7 | `Destination` | Categorical | Travel destination country |
| 8 | `Net Sales` | Numerical | Net sales value (local currency) |
| 9 | `Commision (in value)` | Numerical | Commission paid to agent |
| 10 | `Age` | Numerical | Policyholder age |
| 11 | `Claim` | Target | Whether a claim was filed: Yes/No |

### Data Cleaning Summary

| Issue | Action | Rationale |
|---|---|---|
| Age = 118 | Replaced with median age | Unrealistic value — likely a placeholder/input error |
| Negative Duration | Converted to absolute value | Trip duration cannot be negative, sign error |
| Gender >30% missing | **Dropped column** | High missing rate + no correlation with claim, Occam's Razor |
| Outliers (Duration, Net Sales) | **Retained** | Natural variation; handled by RobustScaler during preprocessing |

### Feature Engineering

| New Feature | Logic | Business Rationale |
|---|---|---|
| `Is_Senior` | Age > 60 = 1 | Senior travelers have higher health risk exposure |
| `Sales_Per_Day` | Net Sales / Duration | Proxy for coverage level per day |
| `Commission_Rate` | Commission / Net Sales | High commission ratio may indicate specific product types |

---

## Analytical Workflow (The Resampling Battle)

### The Challenge

With a 1:64 imbalance ratio, a naive model that predicts "No Claim" for everyone achieves 98.5% accuracy but catches zero actual claims. This is useless. We needed aggressive resampling to teach the model what a "Claim" looks like.

### Experimental Design

We tested 6 algorithms x 4 resampling scenarios = 24 combinations using 5-Fold Stratified Cross-Validation with F2-Score as the evaluation metric.

**Algorithms Tested:**

| Model | Selection Rationale |
|---|---|
| Logistic Regression | Linear baseline: fast, interpretable |
| Decision Tree | Non-linear baseline: captures feature interactions |
| Random Forest | Bagging ensemble: reduces overfitting from single trees |
| Gradient Boosting | Sequential boosting: strong at correcting prior errors |
| XGBoost | Optimized boosting: native sparse handling, L1/L2 regularization |
| LightGBM | Fast boosting: histogram-based splitting, efficient for high-cardinality categoricals |

**Resampling Strategies:**

| Strategy | Technique | How It Works |
|---|---|---|
| No Resampling | Baseline | Train on original imbalanced data |
| Over Sampling | `RandomOverSampler` | Duplicate minority samples randomly |
| Under Sampling | `RandomUnderSampler` | Reduce majority samples randomly |
| Hybrid | `SMOTE + Tomek Links` | SMOTE creates synthetic samples + Tomek cleans decision boundaries |

### Final Model Selected

```
Algorithm  : Gradient Boosting
Resampling : SMOTE + Tomek Links
Tuning     : RandomizedSearchCV (50 iterations, 5-Fold CV)
```

---

## Evaluation Metrics & Optimization

### Why F2-Score?

The F2-Score gives 2x weight to Recall over Precision. This means the model is penalized more heavily for missing real claims (FN) than for false alarms (FP). This is aligned with our business priority of catching every possible claim.

**Why not pure Recall?** Maximizing Recall alone would push the model to predict *everyone* as "Claim", 100% Recall but ~0% Precision, which is operationally useless. F2-Score ensures balance while still prioritizing Recall.

### Profit-Based Threshold Optimization

Beyond the standard F2-Score threshold, we implemented profit-based threshold optimization. Searching for the classification threshold that maximizes total financial savings for Global-Guard.

```
Standard threshold (F2-optimized) : 0.59
Profit-optimized threshold        : 0.82
```

**The "Sweet Spot":** By tuning the threshold to 0.82, we found the point where the model achieves the best trade-off between catching claims and minimizing false alarm costs, maximizing the actual dollar savings for the company.

### Final Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| **F2-Score** | 0.228 | Primary metric — weighted towards catching claims |
| **Recall** | 34.1% | Proportion of actual claims detected |
| **Precision** | 9.8% | Proportion of predicted claims that are real |
| **ROC-AUC** | 0.832 | Excellent discrimination ability across all thresholds |
| **Profit Savings** | **$930** | Net savings vs. no-model baseline (on test set) |
| **ROI** | **1.38%** | Return on investment from model deployment |

**Important Context:** The extreme class imbalance (1:64) makes raw F2/Recall numbers appear low, but the **ROC-AUC of 0.832** confirms that the model has strong discriminative ability. The profit-optimized threshold prioritizes financial outcomes over raw metric scores.

---

## Key Insights (SHAP Analysis)

SHAP (SHapley Additive exPlanations) analysis reveals which features **most influence** the model's claim prediction:

### Top Predictive Features

1. **Product Name**: Certain insurance products have inherently higher claim rates. Products with broader coverage naturally attract more claims.

2. **Agency**: Different agencies serve different customer segments. Agencies specializing in high-risk travel corridors show elevated claim rates.

3. **Destination**: Travel destination is a strong risk indicator. Regions with higher health/safety risks correlate with more claims.

4. **Duration**: Longer trips = longer exposure to risk. Trips exceeding 90 days show disproportionately higher claim rates.

5. **Net Sales**: Higher-value policies often have more comprehensive coverage, making it easier for policyholders to meet claim eligibility criteria.

6. **Commission Rate**: Products with higher commission rates tend to be pushed more aggressively by agents, sometimes to customers who may not fully understand the coverage, leading to mismatched expectations and disputes.

These insights directly inform the business recommendations below. They're not just statistical curiosities, they're actionable intelligence for the Operations and Marketing teams.

---

## Business Recommendations

### For the Operations Team

1. **Risk-Based Reserve Allocation**
   - Use model scores to dynamically allocate claim reserves per policy
   - High-risk scores = higher reserve provisions; low-risk = leaner reserves

2. **Agency Audit Program**
   - Agencies with persistently high claim rates should be audited for adverse selection or potential fraud
   - Implement quarterly performance reviews with claim rate KPIs

3. **Destination Risk Tiers**
   - Create a tiered risk scoring system based on destination
   - High-risk destinations should trigger supplementary underwriting checks

4. **Duration-Based Assessment**
   - Trips exceeding 90 days should require additional risk assessment documentation
   - Consider premium adjustments for extended travel durations

### For the Marketing & Product Team

5. **Product Repricing**
   - Products with claim rates significantly above average should be repriced to reflect their true risk
   - Consider redesigning high-claim products with tighter coverage terms

6. **Distribution Channel Strategy**
   - Analyze whether Online vs. Offline channels attract different risk profiles
   - Tailor underwriting requirements per channel if significant differences exist

### For the Technology Team

7. **Model Monitoring & Retraining**
   - Retrain the model every 6-12 months with fresh data to prevent concept drift
   - Implement A/B testing before full production deployment
   - Monitor for data quality issues that could degrade predictions

---

## How to Use

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn
```

### Loading the Saved Model

```python
import pickle
import json
import pandas as pd

# Load the trained model pipeline
model = pickle.load(open('models/final_model_travel_insurance.sav', 'rb'))

# Load model metadata (features, thresholds, metrics)
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Model: {metadata['model_name']}")
print(f"Resampling: {metadata['resampling']}")
print(f"Optimal F2 Threshold: {metadata['f2_threshold']}")
print(f"Profit Threshold: {metadata['profit_threshold']}")
```

### Making Predictions

```python
# Example: predict on new data
new_policy = pd.DataFrame({
    'Agency': ['EPX'],
    'Agency Type': ['Travel Agency'],
    'Distribution Channel': ['Online'],
    'Product Name': ['Comprehensive Plan'],
    'Duration': [15],
    'Destination': ['SINGAPORE'],
    'Net Sales': [50.0],
    'Commision (in value)': [10.0],
    'Age': [35],
    'Is_Senior': [0],
    'Sales_Per_Day': [3.33],
    'Commission_Rate': [0.20]
})

# Get probability
prob = model.predict_proba(new_policy)[:, 1]

# Apply profit-optimized threshold
threshold = metadata['profit_threshold']  # 0.82
prediction = (prob >= threshold).astype(int)

print(f"Claim Probability: {prob[0]:.2%}")
print(f"Prediction (threshold={threshold}): {'CLAIM' if prediction[0] else 'NO CLAIM'}")
```

### Running the Full Notebook

```bash
# 1. Clone the repository
git clone https://github.com/ravasrgh/travel-insurance-claim-prediction.git
cd travel-insurance-claim-prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn statsmodels

# 3. Open and run the notebook
jupyter notebook TravelInsurance_CapstoneProject3.ipynb

# Or open in VS Code with the Jupyter extension and click "Run All"
# Expected runtime: ~10-15 minutes (ML benchmarking is compute-intensive)
```

---

## Conclusion

### What We Built

A machine learning-powered early warning system for travel insurance claims that enables Global-Guard Travel Assurance to proactively allocate claim reserves instead of reacting to unexpected claims.

### Key Results

| Metric | Value |
|---|---|
| Best Model | Gradient Boosting + SMOTE |
| ROC-AUC | 0.832 (strong discriminative ability) |
| F2-Score | 0.228 (on extremely imbalanced data, 1:64) |
| Net Savings | $930 on test set |
| ROI | 1.38% positive return |

### Financial Impact

```
Without Model:  All claims are unexpected = Full emergency response cost
With Model:     Claims are anticipated = Proactive, cost-efficient handling

Savings on test set alone: $930
Annualized (extrapolated to full portfolio): Significant reserve optimization
```

### Limitations & Future Work

| Limitation | Proposed Solution |
|---|---|
| No temporal data (policy dates) | Collect timestamps for seasonal trend analysis |
| Extreme imbalance (1:64) limits Recall | Explore cost-sensitive learning, focal loss |
| Simplified cost assumptions | Calibrate with actual claim payout data from Finance |
| Static model | Implement quarterly retraining pipeline |
| Single-model approach | Explore stacking ensembles for improved performance |

---

## Project Structure

```
travel-insurance-claim-prediction/
|
|-- README.md                                      # This file
|-- TravelInsurance_CapstoneProject3.ipynb          # Main analysis notebook
|-- data_travel_insurance.csv                       # Raw dataset (44,328 records)
|
|-- models/
    |-- final_model_travel_insurance.sav            # Trained model (Pickle)
    |-- model_metadata.json                         # Model info, thresholds, metrics
```


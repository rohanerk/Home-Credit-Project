# Home Credit Default Risk — Credit Intelligence Project

## Business Problem
A consumer lender wants to grow its loan book without raising its default rate.
This project asks: **which applicants is the lender currently mispricing, what
approval policy maximises expected profit, and does that policy treat all
demographic groups fairly?**

---

## Project Structure

    Home Credit Project/
    ├── notebooks/
    │   └── home_credit_analysis.ipynb
    ├── data/
    ├── outputs/
    │   ├── home_credit.duckdb
    │   ├── eda_findings.png
    │   ├── roc_curve.png
    │   ├── feature_importance.png
    │   ├── shap_summary.png
    │   ├── shap_waterfall.png
    │   ├── policy_simulation.png
    │   ├── fairness_audit.png
    │   └── iv_chart.png
    ├── app.py
    └── README.md

---

## Methodology

### 1. Data Engineering
- Loaded 8 CSV files (58 million rows) into DuckDB
- Built an applicant-level feature mart entirely in SQL using CTEs:
  - `bureau` — prior credits at other institutions
  - `previous_application` — prior Home Credit loan history
  - `installments_payments` — payment behaviour on prior loans
- Output: 307,511 rows × 40 features, one row per applicant
- Data quality fix: `DAYS_EMPLOYED = 365243` converted to NULL

### 2. Exploratory Analysis — 5 Business Findings
1. **Thin-file customers default more** — 10.1% vs 7.7% for customers with credit history
2. **Late payments predict default** — 6.7% bad rate for clean payers vs 9.8% for chronic late payers
3. **Income type matters** — pensioners (5.4%) are safest; working-class (9.6%) highest risk
4. **Credit-to-income ratio** — default rate rises from 6.4% at less than 1x to 8.8% at 3–5x income
5. **EXT_SOURCE_2 dominates** — cleanly separates defaulters from non-defaulters

### 3. Predictive Model
- **Algorithm:** HistGradientBoostingClassifier (scikit-learn)
- **Handles missing values natively** — no imputation required
- **Validation AUC:** 0.7521
- **Top predictors:** EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_1, employment_years, prev_refused_count
- **Class imbalance:** handled via class_weight='balanced' (8% default rate)

### 4. SHAP Explanations
- Calculated SHAP values on a 2,000-applicant sample
- **Summary plot:** EXT_SOURCE_2 and EXT_SOURCE_3 dominate — high scores reduce predicted default probability
- **Waterfall plot:** A 74.2% probability defaulter was flagged due to low EXT_SOURCE_3 (0.185), missing EXT_SOURCE_1, high bureau debt (1.66M), and 7 late payments
- **Governance concern:** Top three predictors are from an undisclosed provider — lender cannot explain rejections to regulators

### 5. Policy Simulation
- Swept approval cut-offs from 0.05 to 0.95
- Applied a board-mandated bad rate ceiling of 5%
- **Recommended cut-off: 0.61**
  - Approves 80.7% of applicants
  - Bad rate: 5.0%
  - Total portfolio profit: 12.23B
- Tightening to 0.55 costs 927M in foregone profit for a 0.6pp bad rate reduction
- Assumptions: Avg loan 500,000 | Margin 18% | Term 3yr | LGD 40%

### 6. Fairness Audit
- **Gender results:**
  - Female: approval 83.1%, bad rate 7.0%
  - Male: approval 76.0%, bad rate 10.2%
  - 80% Rule ratio: 0.915 — PASS (threshold: 0.80)
- **Income type:** approval rate differences are proportional to actual default rate differences — model is well-calibrated
- Note: 80% rule is a US fair lending standard used here as a reference framework

### 7. WoE/IV Scorecard
- Built an interpretable logistic regression scorecard excluding EXT_SOURCE scores
- Only 1 feature (employment_years) achieved Medium IV (0.116) — all others Weak
- **Scorecard AUC: 0.6624 vs HistGBM AUC: 0.7521**
- The 0.09 AUC gap is almost entirely explained by EXT_SOURCE opacity
- **Recommendation:** Champion-challenger architecture — scorecard as primary decision tool, HistGBM as challenger

### 8. Streamlit App
- Built an interactive 3-tab dashboard for credit officers:
  - **Score Explorer** — score distribution by actual outcome
  - **Policy Simulator** — live sliders for cut-off, loan amount, margin, LGD, bad rate ceiling
  - **Fairness Monitor** — real-time gender and income type fairness audit at any cut-off

---

## Key Limitations
1. **Reject inference bias** — trained only on approved applicants; may underestimate risk for rejected profiles
2. **EXT_SOURCE opacity** — top predictors from undisclosed provider; regulatory risk in governed markets
3. **Temporal validity** — single time period; performance may degrade across economic cycles
4. **Assumed business parameters** — profit figures sensitive to loan size, margin, and LGD assumptions

---

## Next Steps
- [ ] Deploy Streamlit app to Streamlit Cloud for public live link
- [ ] Fairness audit extended to education and family status segments
- [ ] Sensitivity analysis across LGD and margin assumptions

---

## Results Summary

| Metric | Value |
|---|---|
| Validation AUC (HistGBM) | 0.7521 |
| Validation AUC (Scorecard) | 0.6624 |
| Recommended cut-off | 0.61 |
| Approval rate | 80.7% |
| Bad rate | 5.0% |
| Portfolio profit | 12.23B |
| Gender 80% rule ratio | 0.915 PASS |
| Interactive app | Streamlit (3 tabs) |

---

## Setup

    py -3.11 -m venv venv311
    venv311\Scripts\activate
    python -m pip install duckdb pandas numpy matplotlib seaborn scikit-learn shap notebook ipykernel streamlit optbinning
    code notebooks/home_credit_analysis.ipynb

    # Run the Streamlit app
    streamlit run app.py

## Dataset
[Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

Download all CSV files and place them in the `data/` folder before running.

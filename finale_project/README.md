# TBML XAI Compliance Dashboard

An interactive Streamlit dashboard for compliance officers reviewing
Trade-Based Money Laundering (TBML) alerts. Instead of a bare probability
score, every flagged transaction gets:

- A **SHAP waterfall plot** showing exactly which features pushed the score
  up or down.
- A **plain-English audit narrative** (downloadable as `.txt`) suitable for
  a case file, e.g.:

  ```
  Transaction TST_002247 was FLAGGED with an estimated fraud probability of 91.4%.

  Primary contributing factors:
    (+) vendor country (Sanctioned_Proxy_Beta) is on the high-risk/sanctioned jurisdiction list (impact: +0.412)
    (+) unit price is 24.3% above the market spot price (impact: +0.287)
    (+) transaction value is only $150.10 away from the $10,000 regulatory reporting threshold (impact: +0.118)
    (-) payment method used was Letter_of_Credit (impact: -0.041)
  ```

- A **LIME cross-check** for a second, independent local explanation.
- A **global SHAP summary plot** for model-risk / validation review of the
  model's overall behavior.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — three tabs: flagged list, per-case explanation, global model behavior |
| `model_utils.py` | Feature engineering + XGBoost training pipeline |
| `explain_utils.py` | SHAP/LIME computation, waterfall/summary plots, narrative-text generation |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data

Place the ledger CSVs (same schema used in the SOC 2026 report) at:

```
data/train_metallurgical_ledgers.csv
data/test_metallurgical_ledgers.csv
```

or upload them via the sidebar at runtime. Required columns:

```
Transaction_ID, Date, Vendor_ID, Vendor_Country, Commodity, Volume_MT,
Market_Spot_Price, Unit_Price_USD, Total_Value_USD, Payment_Method,
Is_Fraud_Ground_Truth, Fraud_Type
```

`Is_Fraud_Ground_Truth` is optional at inference time — if absent, the
dashboard still scores and explains transactions, it just can't show
accuracy/precision/recall on the "Global Model Behavior" tab.

## Run

```bash
streamlit run app.py
```

## How it fits the "6 models" report

This dashboard trains a 7th model — an XGBoost classifier on a mix of raw
features *and* the same engineered rule signals from Models 1–3 in the
original report (price deviation %, structuring-band flag, high-risk-country
flag) plus a couple of extra ratios (log transaction value, value-per-MT).
Feeding SHAP these business-meaningful engineered features — rather than only
raw one-hot columns — is what makes the auto-generated narratives read like
something a human investigator would write, instead of a list of opaque
feature indices.

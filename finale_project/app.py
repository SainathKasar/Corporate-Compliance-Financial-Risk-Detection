"""
app.py
XAI Compliance Dashboard for Trade-Based Money Laundering (TBML) detection.

Run with:
    streamlit run app.py

Expects the two CSVs from the Seasons of Code project either at:
    ./data/train_metallurgical_ledgers.csv
    ./data/test_metallurgical_ledgers.csv
or uploaded via the sidebar at runtime.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from model_utils import train_model, transform_for_shap, predict_proba, FEATURE_ORDER
from explain_utils import (
    compute_shap_values,
    top_contributors,
    generate_narrative,
    plot_waterfall,
    plot_global_summary,
    build_lime_explainer,
    explain_with_lime,
)

st.set_page_config(page_title="TBML XAI Compliance Dashboard", layout="wide")

DEFAULT_TRAIN_PATH = "data/train_metallurgical_ledgers.csv"
DEFAULT_TEST_PATH = "data/test_metallurgical_ledgers.csv"


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


@st.cache_resource(show_spinner="Training XGBoost model...")
def get_trained_model(train_df: pd.DataFrame):
    pipeline, preprocessor = train_model(train_df)
    return pipeline


@st.cache_resource(show_spinner="Computing SHAP values for test set...")
def get_shap_values(_pipeline, test_df: pd.DataFrame):
    X_t, eng_df, feature_names = transform_for_shap(_pipeline, test_df)
    shap_values, explainer = compute_shap_values(_pipeline, X_t)
    return shap_values, X_t, eng_df, feature_names


@st.cache_resource(show_spinner=False)
def get_lime(_pipeline, X_train_transformed, feature_names):
    return build_lime_explainer(X_train_transformed, feature_names)


# ---------------------------------------------------------------------------
# Sidebar: data source + threshold
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

train_file = st.sidebar.file_uploader("Training ledger CSV", type="csv")
test_file = st.sidebar.file_uploader("Test / live ledger CSV", type="csv")

train_path = train_file if train_file is not None else (
    DEFAULT_TRAIN_PATH if os.path.exists(DEFAULT_TRAIN_PATH) else None
)
test_path = test_file if test_file is not None else (
    DEFAULT_TEST_PATH if os.path.exists(DEFAULT_TEST_PATH) else None
)

if train_path is None or test_path is None:
    st.title("🔎 TBML Explainable AI Compliance Dashboard")
    st.warning(
        "Upload a training ledger and a test/live ledger CSV in the sidebar "
        "(or place them at `data/train_metallurgical_ledgers.csv` and "
        "`data/test_metallurgical_ledgers.csv`) to get started."
    )
    st.stop()

train_df = load_csv(train_path)
test_df = load_csv(test_path)

threshold = st.sidebar.slider("Fraud probability flag threshold", 0.0, 1.0, 0.5, 0.01)
top_n_factors = st.sidebar.slider("Top contributing factors to display", 3, 10, 5)

# ---------------------------------------------------------------------------
# Train model + compute explanations (cached)
# ---------------------------------------------------------------------------
pipeline = get_trained_model(train_df)
shap_values, X_test_t, eng_test_df, feature_names = get_shap_values(pipeline, test_df)

fraud_proba = predict_proba(pipeline, test_df)
eng_test_df = eng_test_df.copy()
eng_test_df["Fraud_Probability"] = fraud_proba
eng_test_df["Flagged"] = eng_test_df["Fraud_Probability"] >= threshold

# ---------------------------------------------------------------------------
# Header / KPIs
# ---------------------------------------------------------------------------
st.title("🔎 TBML Explainable AI Compliance Dashboard")
st.caption(
    "XGBoost fraud model + SHAP/LIME explanations for trade-based money "
    "laundering detection. Every flag comes with a plain-English audit trail."
)

n_flagged = int(eng_test_df["Flagged"].sum())
n_total = len(eng_test_df)
if "Is_Fraud_Ground_Truth" in eng_test_df.columns:
    n_true_frauds_caught = int(
        eng_test_df.loc[eng_test_df["Flagged"], "Is_Fraud_Ground_Truth"].sum()
    )
else:
    n_true_frauds_caught = None

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Transactions evaluated", f"{n_total:,}")
kpi2.metric("Flagged for review", f"{n_flagged:,}")
kpi3.metric("Flag rate", f"{n_flagged / n_total:.1%}")
if n_true_frauds_caught is not None:
    kpi4.metric("Confirmed frauds among flagged", f"{n_true_frauds_caught:,}")

tab_overview, tab_case, tab_global = st.tabs(
    ["📋 Flagged Transactions", "🧾 Case Explanation", "🌐 Global Model Behavior"]
)

# ---------------------------------------------------------------------------
# Tab 1: Flagged transaction list
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Transactions flagged for compliance review")
    display_cols = [
        "Transaction_ID", "Date", "Vendor_Country", "Commodity", "Payment_Method",
        "Total_Value_USD", "Unit_Price_USD", "Market_Spot_Price", "Fraud_Probability",
    ]
    display_cols = [c for c in display_cols if c in eng_test_df.columns]

    flagged_view = (
        eng_test_df[eng_test_df["Flagged"]][display_cols]
        .sort_values("Fraud_Probability", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        flagged_view.style.format({"Fraud_Probability": "{:.1%}", "Total_Value_USD": "${:,.2f}"}),
        use_container_width=True,
        height=450,
    )
    st.caption(
        f"Showing {len(flagged_view):,} of {n_total:,} transactions at threshold "
        f"{threshold:.0%}. Select a Transaction_ID in the 'Case Explanation' tab to audit."
    )

# ---------------------------------------------------------------------------
# Tab 2: Single-transaction explanation (SHAP waterfall + narrative + LIME)
# ---------------------------------------------------------------------------
with tab_case:
    st.subheader("Per-transaction audit trail")

    id_col = "Transaction_ID" if "Transaction_ID" in eng_test_df.columns else eng_test_df.index.name
    options = eng_test_df.sort_values("Fraud_Probability", ascending=False)[id_col].tolist()
    selected_id = st.selectbox("Select a transaction to explain", options)

    row_idx = eng_test_df.index[eng_test_df[id_col] == selected_id][0]
    row = eng_test_df.loc[row_idx]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Fraud probability", f"{row['Fraud_Probability']:.1%}")
        st.write(f"**Vendor country:** {row.get('Vendor_Country', 'n/a')}")
        st.write(f"**Commodity:** {row.get('Commodity', 'n/a')}")
        st.write(f"**Payment method:** {row.get('Payment_Method', 'n/a')}")
        st.write(f"**Total value:** ${row.get('Total_Value_USD', 0):,.2f}")
        if "Is_Fraud_Ground_Truth" in row:
            st.write(f"**Ground truth label:** {'Fraud' if row['Is_Fraud_Ground_Truth'] == 1 else 'Legitimate'}")

    contributors = top_contributors(
        shap_values.values[row_idx], feature_names, row, top_n=top_n_factors
    )

    with c2:
        st.markdown("**SHAP waterfall — why this score?**")
        fig = plot_waterfall(shap_values[row_idx], max_display=top_n_factors + 2)
        st.pyplot(fig, clear_figure=True)

    st.markdown("### 📝 Plain-English audit narrative")
    narrative = generate_narrative(str(selected_id), row["Fraud_Probability"], contributors)
    st.code(narrative, language=None)

    st.download_button(
        "Download narrative (.txt)",
        data=narrative,
        file_name=f"{selected_id}_explanation.txt",
        mime="text/plain",
    )

    with st.expander("Cross-check with LIME (independent local explainer)"):
        X_train_t, _, _ = transform_for_shap(pipeline, train_df)
        lime_explainer = get_lime(pipeline, X_train_t, feature_names)
        lime_exp = explain_with_lime(lime_explainer, pipeline, X_test_t[row_idx], num_features=top_n_factors)
        lime_df = pd.DataFrame(lime_exp.as_list(), columns=["Condition", "Weight"])
        st.dataframe(lime_df, use_container_width=True)
        st.caption(
            "LIME perturbs the transaction locally and fits a simple linear model "
            "to approximate the XGBoost decision boundary near this point — a useful "
            "sanity check against the SHAP explanation above."
        )

# ---------------------------------------------------------------------------
# Tab 3: Global model behavior (for model risk / validation review)
# ---------------------------------------------------------------------------
with tab_global:
    st.subheader("Global feature importance across the test set")
    st.caption(
        "For model validation and periodic model-risk review: which factors "
        "drive the model's decisions on average, across all evaluated transactions."
    )
    fig_global = plot_global_summary(shap_values.values, X_test_t, feature_names, max_display=15)
    st.pyplot(fig_global, clear_figure=True)

    st.markdown("---")
    st.subheader("Model performance on this test set")
    if "Is_Fraud_Ground_Truth" in eng_test_df.columns:
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        y_true = eng_test_df["Is_Fraud_Ground_Truth"]
        y_pred = eng_test_df["Flagged"].astype(int)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
        m2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
        m3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
        m4.metric("F1-Score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
    else:
        st.info("No ground-truth labels in this dataset — performance metrics unavailable.")

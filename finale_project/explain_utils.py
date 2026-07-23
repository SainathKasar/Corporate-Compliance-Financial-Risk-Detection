"""
explain_utils.py
Turns raw SHAP values into (a) matplotlib plots and (b) plain-English audit
narratives that a compliance officer can drop straight into a case file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Human-readable labels / phrasing for each engineered or raw feature.
# Keyed by the *prefix* of the transformed feature name, since one-hot
# columns look like "Vendor_Country_Sanctioned_Proxy_Beta".
# ---------------------------------------------------------------------------
FEATURE_PHRASES = {
    "Price_Deviation_Pct": lambda row: (
        f"unit price is {row['Price_Deviation_Pct']:.1f}% "
        f"{'above' if row['Price_Deviation_Pct'] >= 0 else 'below'} the market spot price"
    ),
    "Is_Overpriced_20pct": lambda row: "unit price exceeds market spot price by more than 20% (over-invoicing pattern)",
    "Distance_From_10k": lambda row: (
        f"transaction value is only ${row['Distance_From_10k']:.2f} away from the "
        f"$10,000 regulatory reporting threshold"
    ),
    "Is_Structuring_Band": lambda row: "transaction value falls just under the $10,000 reporting threshold (possible structuring)",
    "Is_High_Risk_Country": lambda row: f"vendor country ({row['Vendor_Country']}) is on the high-risk/sanctioned jurisdiction list",
    "Log_Total_Value": lambda row: f"transaction size (${row['Total_Value_USD']:,.2f}) influenced the score",
    "Value_Per_MT": lambda row: f"value-per-metric-ton (${row['Value_Per_MT']:,.2f}/MT) is atypical for this commodity",
    "Volume_MT": lambda row: f"shipment volume ({row['Volume_MT']:.1f} MT) influenced the score",
    "Market_Spot_Price": lambda row: f"prevailing market spot price (${row['Market_Spot_Price']:.2f}) influenced the score",
    "Unit_Price_USD": lambda row: f"documented unit price (${row['Unit_Price_USD']:.2f}) influenced the score",
    "Total_Value_USD": lambda row: f"total declared value (${row['Total_Value_USD']:,.2f}) influenced the score",
    "Vendor_Country": lambda row: f"vendor country is {row['Vendor_Country']}",
    "Commodity": lambda row: f"commodity type is {row['Commodity']}",
    "Payment_Method": lambda row: f"payment method used was {row['Payment_Method']}",
}


def _phrase_for_feature(feature_name: str, row: pd.Series) -> str:
    """Match a transformed/one-hot feature name back to a human phrase."""
    for prefix, fn in FEATURE_PHRASES.items():
        if feature_name == prefix or feature_name.startswith(prefix + "_"):
            try:
                return fn(row)
            except Exception:
                break
    # Fallback: just clean up the raw name
    return feature_name.replace("_", " ")


def _legacy_base_score_workaround(classifier):
    """Legacy fallback for OLDER shap builds that parse a booster's
    `base_score` with a bare `float(...)` call and choke on XGBoost's
    JSON-array serialization (e.g. '[5E-1]'), raising:
        ValueError: could not convert string to float: '[5E-1]'
    Re-saving the booster's raw bytes (legacy binary format, stripped of the
    4-byte version header) sidesteps that parser.

    IMPORTANT: this must NOT be applied unconditionally. Newer shap
    (>=0.45-ish) loads XGBoost models via UBJSON and calls
    `booster.save_raw(raw_format="ubj")` internally — if `save_raw` has been
    monkeypatched here to always return legacy-format bytes regardless of
    `raw_format`, the UBJSON decoder gets corrupted input and fails with
    something like `ValueError: Expected type size for b'\\x00' but could not
    find any.` So this is only ever called as a fallback after the normal
    path has already failed with the specific old-style error.
    """
    booster = classifier.get_booster()
    raw = booster.save_raw()[4:]
    booster.save_raw = lambda *args, **kwargs: raw


def compute_shap_values(pipeline, X_transformed: np.ndarray):
    """Builds a TreeExplainer for the fitted XGBoost step and returns a
    shap.Explanation object over the already-transformed matrix."""
    classifier = pipeline.named_steps["classifier"]

    try:
        # Normal path: works for modern shap/xgboost combos (UBJSON-based).
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_transformed)
        return shap_values, explainer
    except ValueError as e:
        msg = str(e)
        if "could not convert string to float" not in msg and "base_score" not in msg:
            raise  # a different, unrelated failure — don't mask it

    # Only reached for the specific legacy base_score bug on older shap.
    _legacy_base_score_workaround(classifier)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_transformed)
    return shap_values, explainer


def top_contributors(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    raw_row: pd.Series,
    top_n: int = 5,
) -> pd.DataFrame:
    """Returns the top_n features driving this single prediction, ranked by
    |SHAP value|, with a human-readable phrase and direction."""
    order = np.argsort(-np.abs(shap_values_row))[:top_n]
    records = []
    for i in order:
        val = shap_values_row[i]
        name = feature_names[i]
        records.append(
            {
                "feature": name,
                "shap_value": val,
                "direction": "increases" if val > 0 else "decreases",
                "explanation": _phrase_for_feature(name, raw_row),
            }
        )
    return pd.DataFrame(records)


def generate_narrative(
    transaction_id: str,
    fraud_probability: float,
    contributors_df: pd.DataFrame,
) -> str:
    """Produces a short, audit-ready paragraph summarizing why a transaction
    was flagged (or cleared)."""
    verdict = "FLAGGED" if fraud_probability >= 0.5 else "NOT flagged"
    lines = [
        f"Transaction {transaction_id} was {verdict} with an estimated fraud "
        f"probability of {fraud_probability:.1%}.",
        "",
        "Primary contributing factors:",
    ]
    increasing = contributors_df[contributors_df["shap_value"] > 0]
    decreasing = contributors_df[contributors_df["shap_value"] < 0]

    for _, r in increasing.iterrows():
        lines.append(f"  (+) {r['explanation']} (impact: +{r['shap_value']:.3f})")
    for _, r in decreasing.iterrows():
        lines.append(f"  (-) {r['explanation']} (impact: {r['shap_value']:.3f})")

    if increasing.empty:
        lines.append("  No individual factor pushed the score strongly toward fraud.")

    return "\n".join(lines)


def plot_waterfall(shap_values_row_explanation, max_display: int = 10):
    """Returns a matplotlib Figure with a SHAP waterfall plot for one
    transaction. `shap_values_row_explanation` is a single-row shap.Explanation
    slice, e.g. shap_values[i]."""
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values_row_explanation, max_display=max_display, show=False)
    plt.tight_layout()
    return fig


def plot_global_summary(shap_values, X_transformed, feature_names, max_display: int = 15):
    """Returns a matplotlib Figure with a global SHAP beeswarm/summary plot
    across the whole (test) set, for the dashboard's model-overview tab."""
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LIME (secondary/alternative local explainer, for cross-checking SHAP)
# ---------------------------------------------------------------------------
def build_lime_explainer(X_train_transformed: np.ndarray, feature_names: list[str], class_names=("Legitimate", "Fraud")):
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
    )


def explain_with_lime(lime_explainer, pipeline, x_row_transformed: np.ndarray, num_features: int = 8):
    """Explains a single already-transformed row using LIME. Returns the
    LIME Explanation object (has .as_list(), .as_pyplot_figure(), etc.)."""
    classifier = pipeline.named_steps["classifier"]

    def predict_fn(x):
        return classifier.predict_proba(x)

    return lime_explainer.explain_instance(
        x_row_transformed, predict_fn, num_features=num_features
    )
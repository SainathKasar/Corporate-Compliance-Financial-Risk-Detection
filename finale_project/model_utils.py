"""
model_utils.py
Feature engineering + XGBoost training pipeline for the TBML XAI Compliance Dashboard.

Design notes
------------
- We deliberately engineer a handful of *domain* features on top of the raw
  columns (price deviation ratio, distance from the $10k structuring threshold,
  a high-risk-country flag). Pure raw-feature models (see Model 5/6 in the
  original report) are accurate but SHAP explanations over one-hot country
  columns / raw dollar amounts are harder for a compliance officer to read.
  Feeding SHAP a few pre-engineered, business-meaningful features makes the
  resulting explanations map directly onto language auditors already use.
- XGBoost is used (not RandomForest/GBM from sklearn) because shap.TreeExplainer
  has fast, exact support for it, and it handles the class imbalance well via
  `scale_pos_weight`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

HIGH_RISK_COUNTRIES = {"Sanctioned_Proxy_Alpha", "Sanctioned_Proxy_Beta"}

RAW_NUMERICAL = ["Volume_MT", "Market_Spot_Price", "Unit_Price_USD", "Total_Value_USD"]
CATEGORICAL = ["Vendor_Country", "Commodity", "Payment_Method"]

ENGINEERED_NUMERICAL = [
    "Price_Deviation_Pct",      # % above/below market spot price
    "Distance_From_10k",        # abs distance from the $10,000 reporting threshold
    "Log_Total_Value",          # log-scaled transaction size
    "Value_Per_MT",             # sanity-check ratio, catches unit inconsistencies
]

BOOLEAN_FLAGS = [
    "Is_High_Risk_Country",
    "Is_Structuring_Band",      # 9,800 <= total <= 10,000
    "Is_Overpriced_20pct",      # unit price > 1.2x spot
]

ALL_NUMERICAL = RAW_NUMERICAL + ENGINEERED_NUMERICAL + BOOLEAN_FLAGS
FEATURE_ORDER = ALL_NUMERICAL + CATEGORICAL  # order fed into ColumnTransformer


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable, business-meaningful columns used for both modeling
    and later explanation text generation. Returns a copy."""
    out = df.copy()

    out["Price_Deviation_Pct"] = (
        (out["Unit_Price_USD"] - out["Market_Spot_Price"]) / out["Market_Spot_Price"]
    ) * 100.0

    out["Distance_From_10k"] = (out["Total_Value_USD"] - 10_000).abs()

    out["Log_Total_Value"] = np.log1p(out["Total_Value_USD"].clip(lower=0))

    out["Value_Per_MT"] = out["Total_Value_USD"] / out["Volume_MT"].replace(0, np.nan)
    out["Value_Per_MT"] = out["Value_Per_MT"].fillna(0)

    out["Is_High_Risk_Country"] = out["Vendor_Country"].isin(HIGH_RISK_COUNTRIES).astype(int)
    out["Is_Structuring_Band"] = (
        (out["Total_Value_USD"] >= 9800) & (out["Total_Value_USD"] <= 10000)
    ).astype(int)
    out["Is_Overpriced_20pct"] = (
        out["Unit_Price_USD"] > 1.2 * out["Market_Spot_Price"]
    ).astype(int)

    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ALL_NUMERICAL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        verbose_feature_names_out=True,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-friendly (but still unique) names for every column the model sees
    post-transform, in the exact order the transformed matrix produces them."""
    raw_names = preprocessor.get_feature_names_out()
    # strip sklearn's "num__" / "cat__" prefixes for readability downstream
    return [n.split("__", 1)[-1] for n in raw_names]


def train_model(
    train_df: pd.DataFrame,
    label_col: str = "Is_Fraud_Ground_Truth",
    random_state: int = 42,
) -> tuple[Pipeline, ColumnTransformer]:
    """Trains an XGBoost classifier on engineered + raw features.
    Returns (fitted sklearn Pipeline, fitted preprocessor for reuse)."""
    df = engineer_features(train_df)
    y = df[label_col]
    X = df[FEATURE_ORDER]

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    preprocessor = build_preprocessor()
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipeline.fit(X, y)
    return pipeline, pipeline.named_steps["preprocessor"]


def transform_for_shap(pipeline: Pipeline, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Prepares a raw dataframe into (dense transformed matrix, engineered df,
    feature names) ready to hand to a SHAP TreeExplainer built on pipeline's
    classifier."""
    eng_df = engineer_features(df)
    X = eng_df[FEATURE_ORDER]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_t = preprocessor.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    feature_names = get_feature_names(preprocessor)
    return X_t, eng_df, feature_names


def predict_proba(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    eng_df = engineer_features(df)
    X = eng_df[FEATURE_ORDER]
    return pipeline.predict_proba(X)[:, 1]

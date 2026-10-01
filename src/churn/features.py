"""Feature engineering and the preprocessing ColumnTransformer."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add churn-driver features derived from the raw Telco columns."""
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=config.TENURE_BINS,
        labels=config.TENURE_LABELS,
        right=False,
    ).astype(str)
    # Average amount billed per month of tenure; tenure 0 is treated as 1 month.
    df["charges_per_tenure_month"] = df["TotalCharges"] / np.maximum(df["tenure"], 1)
    df["num_services"] = (df[config.SERVICE_COLUMNS] == "Yes").sum(axis=1)
    df["contract_months"] = df["Contract"].map(config.CONTRACT_MONTHS)
    df["auto_payment"] = np.where(
        df["PaymentMethod"].isin(config.AUTOMATIC_PAYMENTS), "Yes", "No"
    )
    return df


def build_preprocessor() -> Pipeline:
    """Feature engineering followed by scaling and one-hot encoding."""
    encode = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features)),
            ("encode", encode),
        ]
    )

"""Tests for data cleaning and feature engineering."""

import pandas as pd

from churn import config
from churn.data import clean
from churn.features import add_features, build_preprocessor


def test_clean_fixes_total_charges_and_target(raw_df: pd.DataFrame) -> None:
    df = clean(raw_df)
    assert df["TotalCharges"].dtype == float
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}
    assert set(df["SeniorCitizen"].unique()) <= {"Yes", "No"}
    assert config.ID_COLUMN not in df.columns


def test_add_features_on_known_rows() -> None:
    df = pd.DataFrame(
        {
            "tenure": [0, 12, 60],
            "TotalCharges": [0.0, 600.0, 3000.0],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)", "Mailed check"],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "OnlineSecurity": ["No", "Yes", "Yes"],
            "OnlineBackup": ["No", "No", "Yes"],
            "DeviceProtection": ["No", "No", "No"],
            "TechSupport": ["No", "No", "Yes"],
            "StreamingTV": ["No internet service", "No", "Yes"],
            "StreamingMovies": ["No internet service", "No", "No"],
        }
    )
    out = add_features(df)

    assert out["tenure_bucket"].tolist() == ["0-12m", "12-24m", "48m+"]
    assert out["charges_per_tenure_month"].tolist() == [0.0, 50.0, 50.0]
    assert out["num_services"].tolist() == [1, 3, 4]
    assert out["contract_months"].tolist() == [1, 12, 24]
    assert out["auto_payment"].tolist() == ["No", "Yes", "No"]
    assert len(df.columns) == 12, "input frame should not be modified"


def test_preprocessor_output_shape(raw_df: pd.DataFrame) -> None:
    X = clean(raw_df)[config.RAW_FEATURES]
    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert not pd.isna(transformed).any()

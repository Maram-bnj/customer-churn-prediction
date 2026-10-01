"""Shared fixtures: a small synthetic dataset with the same columns as the Telco CSV."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

YES_NO = ["Yes", "No"]
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def make_synthetic_telco(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Random customers in the raw Telco format, with a churn signal on contract and tenure."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = (monthly * tenure).round(2).astype(str)
    total[tenure == 0] = " "  # mimics the blank values in the real file
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    phone = rng.choice(YES_NO, n, p=[0.9, 0.1])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.2, 0.25])

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(YES_NO, n),
            "Dependents": rng.choice(YES_NO, n),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(YES_NO, n)),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(YES_NO, n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
    )
    for col in INTERNET_ADDONS:
        df[col] = np.where(internet == "No", "No internet service", rng.choice(YES_NO, n))

    logit = -1.0 + 1.5 * (contract == "Month-to-month") - 0.04 * tenure
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(churn, "Yes", "No")
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_synthetic_telco()


@pytest.fixture
def raw_csv(tmp_path: Path, raw_df: pd.DataFrame) -> Path:
    path = tmp_path / "telco.csv"
    raw_df.to_csv(path, index=False)
    return path

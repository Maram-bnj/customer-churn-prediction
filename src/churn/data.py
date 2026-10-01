"""Loading, cleaning and splitting the Telco dataset."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

logger = logging.getLogger(__name__)


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw Telco CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and encode the target.

    TotalCharges is stored as text and is blank for customers with tenure 0
    (they have not been billed yet), so those values become 0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(df["TotalCharges"].isna().sum())
    if n_missing:
        logger.info("Filling %d blank TotalCharges values with 0", n_missing)
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    # SeniorCitizen is the only 0/1 column; align it with the other Yes/No flags.
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
    if config.TARGET in df.columns:
        df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df.drop(columns=[config.ID_COLUMN], errors="ignore")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate raw features from the target."""
    return df[config.RAW_FEATURES], df[config.TARGET]


def train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with the project's fixed seed."""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

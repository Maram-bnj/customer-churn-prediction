"""Project-wide constants: paths, column names and training settings."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILENAME = "churn_model.joblib"
METRICS_FILENAME = "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

ID_COLUMN = "customerID"
TARGET = "Churn"

# Raw input columns, as they appear in the Telco CSV (minus ID and target).
RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

# Add-on services counted in the "num_services" feature.
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Tenure buckets in months: [0, 12), [12, 24), [24, 48), [48, inf).
TENURE_BINS = [0, 12, 24, 48, float("inf")]
TENURE_LABELS = ["0-12m", "12-24m", "24-48m", "48m+"]

CONTRACT_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}
AUTOMATIC_PAYMENTS = {"Bank transfer (automatic)", "Credit card (automatic)"}

# Columns fed to the preprocessor after feature engineering.
NUMERIC_FEATURES = RAW_NUMERIC + [
    "charges_per_tenure_month",
    "num_services",
    "contract_months",
]
CATEGORICAL_FEATURES = RAW_CATEGORICAL + ["tenure_bucket", "auto_payment"]

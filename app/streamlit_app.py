"""Streamlit app: enter a customer profile and get a churn probability.

Run with: streamlit run app/streamlit_app.py
"""

import pandas as pd
import streamlit as st

from churn import config
from churn.predict import ModelBundle, load_model, predict_customer, top_drivers

YES_NO = ["No", "Yes"]
ADDON_OPTIONS = ["No", "Yes", "No internet service"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
ADDON_LABELS = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}


@st.cache_resource
def get_model() -> ModelBundle:
    return load_model(config.MODELS_DIR / config.MODEL_FILENAME)


def customer_form() -> dict | None:
    """Render the input form; returns the profile once submitted."""
    with st.form("customer"):
        st.subheader("Account")
        c1, c2, c3 = st.columns(3)
        tenure = c1.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly = c2.number_input("Monthly charges ($)", min_value=0.0, value=70.0, step=5.0)
        total = c3.number_input("Total charges ($)", min_value=0.0, value=840.0, step=50.0)
        contract = c1.selectbox("Contract", list(config.CONTRACT_MONTHS))
        payment = c2.selectbox("Payment method", PAYMENT_METHODS)
        paperless = c3.selectbox("Paperless billing", YES_NO, index=1)

        st.subheader("Customer")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", YES_NO)
        partner = c3.selectbox("Partner", YES_NO)
        dependents = c4.selectbox("Dependents", YES_NO)

        st.subheader("Services")
        c1, c2, c3 = st.columns(3)
        phone = c1.selectbox("Phone service", YES_NO, index=1)
        lines = c2.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = c3.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        columns = (c1, c2, c3)
        addons = {
            col: columns[i % 3].selectbox(label, ADDON_OPTIONS)
            for i, (col, label) in enumerate(ADDON_LABELS.items())
        }

        if not st.form_submit_button("Predict churn"):
            return None

    return {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        **addons,
    }


def main() -> None:
    st.set_page_config(page_title="Churn Prediction", layout="wide")
    st.title("Customer churn prediction")
    st.caption("Fill in a customer profile to estimate how likely they are to leave.")

    try:
        bundle = get_model()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    profile = customer_form()
    if profile is not None:
        proba, at_risk = predict_customer(bundle, profile)
        threshold = bundle["threshold"]
        st.metric("Churn probability", f"{proba:.0%}")
        if at_risk:
            st.warning(f"Above the decision threshold ({threshold:.0%}): likely to churn.")
        else:
            st.success(f"Below the decision threshold ({threshold:.0%}): likely to stay.")

    st.subheader(f"Top churn drivers ({bundle['model_name']})")
    st.caption("Global feature importance of the trained model, not specific to this customer.")
    drivers = top_drivers(bundle, n=10)
    st.bar_chart(pd.DataFrame({"importance": drivers}), horizontal=True)


main()


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "fraud_detection_models.pkl"
DATA_PATH = BASE_DIR / "creditcard.csv"

st.title("💳 Financial Fraud Detection")

st.write(
    "Enter the transaction amount to check whether "
    "the transaction is legitimate or fraudulent."
)

# =====================================
# 2. CHECK FILES
# =====================================

if not MODEL_PATH.exists():
    st.error("Model file not found: fraud_detection_models.pkl")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset file not found: creditcard.csv")
    st.stop()

# =====================================
# 3. LOAD TRAINED MODELS
# =====================================

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    models = load_models()
    data = load_data()

    xgb_model = models["xgb_model"]
    iso_model = models["iso_model"]

    xgb_scaler = models["xgb_scaler"]
    imputer = models["imputer"]
    iso_score_scaler = models["iso_score_scaler"]

    xgb_features = models["xgb_features"]
    anomaly_features = models["anomaly_features"]

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# =====================================
# 4. TRANSACTION AMOUNT INPUT
# =====================================

amount = st.number_input(
    "Transaction Amount",
    min_value=0.01,
    value=100.0,
    step=1.0
)

# =====================================
# 5. PREDICTION
# =====================================

if st.button("Check Transaction"):

    try:
        # Find the closest transaction in the dataset
        amount_difference = (
            data["Amount"].astype(float) - float(amount)
        ).abs()

        closest_index = amount_difference.idxmin()

        transaction = data.loc[[closest_index]].copy()

        original_dataset_amount = float(
            transaction["Amount"].iloc[0]
        )

        # Replace dataset amount with entered amount
        transaction["Amount"] = float(amount)

        # Feature engineering
        transaction["Amount_log"] = np.log1p(
            transaction["Amount"].astype(float)
        )

        # =====================================
        # 6. XGBOOST PREDICTION
        # =====================================

        xgb_input = transaction[xgb_features]

        xgb_input_scaled = xgb_scaler.transform(
            xgb_input
        )

        fraud_probability = xgb_model.predict_proba(
            xgb_input_scaled
        )[0][1]

        # =====================================
        # 7. ISOLATION FOREST PREDICTION
        # =====================================

        anomaly_input = transaction[anomaly_features]

        anomaly_input_imputed = imputer.transform(
            anomaly_input
        )

        raw_anomaly_score = -iso_model.decision_function(
            anomaly_input_imputed
        )

        iso_normalized_score = iso_score_scaler.transform(
            raw_anomaly_score.reshape(-1, 1)
        )[0][0]

        # =====================================
        # 8. HYBRID MODEL
        # =====================================

        hybrid_score = (
            0.5 * fraud_probability
            + 0.5 * iso_normalized_score
        )

        hybrid_score = np.clip(
            hybrid_score, 0, 1
        )

        # =====================================
        # 9. FINAL RESULT
        # =====================================

        st.divider()

        st.subheader("Prediction Result")

        if hybrid_score >= 0.50:

            st.error("🚨 FRAUDULENT TRANSACTION")

        else:

            st.success("✅ LEGITIMATE TRANSACTION")

        st.write(f"**Entered Amount:** ₹{amount:,.2f}")

        st.write(
            f"**Closest Dataset Amount:** "
            f"₹{original_dataset_amount:,.2f}"
        )

        st.write(
            f"**Fraud Probability:** "
            f"{fraud_probability * 100:.2f}%"
        )

        st.write(
            f"**Hybrid Risk Score:** "
            f"{hybrid_score * 100:.2f}%"
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")

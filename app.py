
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# =========================================
# LOAD MODELS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

fraud_model = joblib.load(
    os.path.join(BASE_DIR, "fraud_model.pkl")
)

logistic_model = joblib.load(
    os.path.join(BASE_DIR, "logistic_model.pkl")
)

isolation_model = joblib.load(
    os.path.join(BASE_DIR, "isolation_model.pkl")
)

features = joblib.load(
    os.path.join(BASE_DIR, "features.pkl")
)

# =========================================
# TITLE
# =========================================

st.title("💳 AI-Powered Credit Card Fraud Detection System")

st.markdown(
    "### Banking Security & Fraud Analytics Dashboard"
)

st.divider()

# =========================================
# SIDEBAR INPUTS
# =========================================

st.sidebar.header("Transaction Details")

input_data = {}

# Default important fields
default_values = {
    "Time": 10000.0,
    "Amount": 12000.0
}

# Add defaults for PCA features
for i in range(1, 29):
    default_values[f"V{i}"] = 0.0

# Dynamic feature inputs
for feature in features:

    default_value = default_values.get(feature, 0.0)

    input_data[feature] = st.sidebar.number_input(
        feature,
        value=float(default_value)
    )

# =========================================
# DATAFRAME
# =========================================

input_df = pd.DataFrame([input_data])

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("🔍 Detect Fraud"):

    # =====================================
    # RANDOM FOREST PREDICTION
    # =====================================

    rf_prediction = fraud_model.predict(input_df)[0]

    rf_probability = (
        fraud_model.predict_proba(input_df)[0][1] * 100
    )

    # =====================================
    # LOGISTIC REGRESSION
    # =====================================

    log_prediction = logistic_model.predict(input_df)[0]

    log_probability = (
        logistic_model.predict_proba(input_df)[0][1] * 100
    )

    # =====================================
    # ISOLATION FOREST
    # =====================================

    iso_prediction = isolation_model.predict(input_df)[0]

    anomaly_status = (
        "Anomaly Detected"
        if iso_prediction == -1
        else "Normal Transaction"
    )

    # =====================================
    # FINAL RISK LEVEL
    # =====================================

    final_probability = max(
        rf_probability,
        log_probability
    )

    if final_probability > 90:
        risk_level = "CRITICAL"

    elif final_probability > 70:
        risk_level = "HIGH"

    elif final_probability > 40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # =====================================
    # RESULTS
    # =====================================

    st.subheader("Fraud Detection Results")

    col1, col2, col3 = st.columns(3)

    # Random Forest
    with col1:

        st.markdown("### 🌲 Random Forest")

        if rf_prediction == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legitimate")

        st.metric(
            "Fraud Probability",
            f"{rf_probability:.2f}%"
        )

    # Logistic Regression
    with col2:

        st.markdown("### 📈 Logistic Regression")

        if log_prediction == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legitimate")

        st.metric(
            "Fraud Probability",
            f"{log_probability:.2f}%"
        )

    # Isolation Forest
    with col3:

        st.markdown("### 🛡 Isolation Forest")

        if iso_prediction == -1:
            st.error("⚠️ Suspicious Activity")
        else:
            st.success("✅ Normal")

        st.info(anomaly_status)

    st.divider()

    # =====================================
    # FINAL OUTPUT
    # =====================================

    st.subheader("Final AI Risk Assessment")

    final_col1, final_col2 = st.columns(2)

    with final_col1:

        st.metric(
            "Final Fraud Probability",
            f"{final_probability:.2f}%"
        )

    with final_col2:

        st.metric(
            "Risk Level",
            risk_level
        )

    st.progress(float(final_probability / 100))

    # =====================================
    # ALERT MESSAGE
    # =====================================

    if risk_level == "CRITICAL":

        st.error(
            "🚨 Immediate Action Required! Possible Financial Fraud."
        )

    elif risk_level == "HIGH":

        st.warning(
            "⚠️ High Risk Transaction Detected."
        )

    elif risk_level == "MEDIUM":

        st.info(
            "🔍 Medium Risk Transaction."
        )

    else:

        st.success(
            "✅ Low Risk Transaction."
        )

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown(
    "Built with ❤️ using Streamlit, Random Forest, Logistic Regression & Isolation Forest"
)


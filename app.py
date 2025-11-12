import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/best_log_reg_model.pkl"),
        "Random Forest": joblib.load("models/best_rf_classifier.pkl"),
        "XGBoost": joblib.load("models/best_xgb_classifier.pkl")
    }
    return models

models = load_models()

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="EMI Eligibility Prediction App", layout="centered")

st.title("🏦 EMI Eligibility Prediction Dashboard")
st.write("Upload your financial data or enter manually to predict EMI eligibility.")

# -----------------------------
# Sidebar model selection
# -----------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

st.sidebar.info(f"Using: **{model_choice}** for prediction")

# -----------------------------
# Data Input Section
# -----------------------------
st.subheader("📋 Enter Applicant Details")

# Example inputs (customize according to your dataset)
gender = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", min_value=18, max_value=80, value=30)
income = st.number_input("Monthly Income (₹)", min_value=1000, value=50000)
expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=15000)
existing_loans = st.number_input("Existing Loan Amount (₹)", min_value=0, value=20000)
credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=700)
employment_type = st.selectbox("Employment Type", ["salaried", "self-employed", "unemployed"])

# -----------------------------
# Prepare Input for Prediction
# -----------------------------
input_data = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "income": income,
    "expenses": expenses,
    "existing_loans": existing_loans,
    "credit_score": credit_score,
    "employment_type": employment_type
}])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict EMI Eligibility"):
    model = models[model_choice]
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted EMI Eligibility: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

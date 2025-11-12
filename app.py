import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_xgb_classifier.pkl")
    return model

model = load_model()

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="EMI Eligibility Prediction", layout="centered")
st.title("💰 EMI Eligibility Prediction App")
st.markdown("### Predict whether a person is Eligible, Not Eligible, or High Risk for EMI approval.")

# -------------------------------
# User input section
# -------------------------------
st.sidebar.header("Input Applicant Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("male", "female"))
    age = st.sidebar.slider("Age", 18, 70, 30)
    income = st.sidebar.number_input("Monthly Income (₹)", min_value=1000, step=500)
    loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=1000, step=500)
    tenure = st.sidebar.slider("Loan Tenure (months)", 6, 84, 12)
    credit_score = st.sidebar.slider("Credit Score", 300, 900, 700)
    existing_loans = st.sidebar.slider("Existing Loans", 0, 10, 1)

    data = {
        "gender": gender,
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "tenure": tenure,
        "credit_score": credit_score,
        "existing_loans": existing_loans
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Applicant Input Data")
st.write(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Eligibility"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted EMI Eligibility: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and XGBoost")



import streamlit as st
import pandas as pd
import joblib

# Load the trained model and column structure
model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn risk")

# Input fields
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict Churn Risk"):
    # Build input row matching training columns, default everything to 0
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    input_data['tenure'] = tenure
    input_data['MonthlyCharges'] = monthly_charges
    input_data['TotalCharges'] = total_charges
    input_data['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
    
    if f'Contract_{contract}' in input_data.columns:
        input_data[f'Contract_{contract}'] = 1
    if f'InternetService_{internet_service}' in input_data.columns:
        input_data[f'InternetService_{internet_service}'] = 1
    if paperless_billing == "Yes" and 'PaperlessBilling_Yes' in input_data.columns:
        input_data['PaperlessBilling_Yes'] = 1
    if f'PaymentMethod_{payment_method}' in input_data.columns:
        input_data[f'PaymentMethod_{payment_method}'] = 1
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f" High Churn Risk — {probability*100:.1f}% probability")
    else:
        st.success(f" Low Churn Risk — {probability*100:.1f}% probability")
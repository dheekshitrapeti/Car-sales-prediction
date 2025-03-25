import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.title("🚗 Car Purchase Prediction App")

# User Input Fields
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.number_input("Enter Age", min_value=18, max_value=100, value=25)
annual_salary = st.number_input("Enter Annual Salary ($)", min_value=0, value=50000)
credit_card_debt = st.number_input("Enter Credit Card Debt ($)", min_value=0, value=5000)
net_worth = st.number_input("Enter Net Worth ($)", min_value=0, value=70000)
country = st.text_input("Enter Country (e.g., India, USA, Canada)")


# Prediction Logic
if st.button("Predict Car Purchase Amount 💰"):
    # Prepare input features
    feature_values = np.array([[age, annual_salary, net_worth]])
    
    # Predict
    prediction = model.predict(feature_values)
    predicted_amount = float(prediction[0])  # Ensure it's a scalar float
    
    # Display user details
    st.write("### User Details")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age}")
    st.write(f"**Annual Salary:** ${annual_salary:,}")
    st.write(f"**Credit Card Debt:** ${credit_card_debt:,}")
    st.write(f"**Net Worth:** ${net_worth:,}")
    st.write(f"**Country:** {country}")
    
    # Display result
    st.success(f"💰 Predicted Car Purchase Amount: ${predicted_amount:,.2f}")

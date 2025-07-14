import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/vehicle_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Vehicle Price Predictor", layout="centered")
st.title("🚗 Vehicle Price Prediction Dashboard")

# Input fields
mileage = st.number_input("Mileage (in miles)", min_value=0, max_value=500000, value=50000)
vehicle_age = st.slider("Vehicle Age", 0, 30, 5)
doors = st.selectbox("Number of Doors", [2, 3, 4, 5, 6])

make = st.selectbox("Make", ['Toyota', 'Ford', 'Honda', 'Unknown'])
model_name = st.selectbox("Model", ['Camry', 'Civic', 'F-150', 'Unknown'])
fuel = st.selectbox("Fuel Type", ['Gasoline', 'Diesel', 'Electric', 'Hybrid', 'Unknown'])
transmission = st.selectbox("Transmission", ['Automatic', 'Manual', 'Unknown'])

# Scale mileage and vehicle_age first
scaled_values = scaler.transform([[mileage, vehicle_age]])
scaled_mileage = scaled_values[0][0]
scaled_vehicle_age = scaled_values[0][1]

# Build input feature dict
input_data = {
    'mileage': scaled_mileage,
    'vehicle_age': scaled_vehicle_age,
    'doors': doors,
    'make_' + make: 1,
    'model_' + model_name: 1,
    'fuel_' + fuel: 1,
    'transmission_' + transmission: 1
}

# Create a full feature set matching training
model_features = model.feature_names_in_
final_input = {feat: input_data.get(feat, 0) for feat in model_features}
X_input = pd.DataFrame([final_input])

# Predict
if st.button("🔮 Predict Vehicle Price"):
    price = model.predict(X_input)[0]
    st.success(f"💰 Estimated Vehicle Price: ${price:,.2f}")

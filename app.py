import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Load model, scaler and feature names
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Title
st.title("🏠 California House Price Predictor")
st.write("Enter the district information below to predict the median house value.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-119.5, format="%.2f")
    latitude = st.number_input("Latitude", value=35.6, format="%.2f")
    housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=52, value=29)
    total_rooms = st.number_input("Total Rooms", min_value=1, value=2100)
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=430)

with col2:
    population = st.number_input("Population", min_value=1, value=1170)
    households = st.number_input("Households", min_value=1, value=400)
    median_income = st.number_input("Median Income (x10k $)", min_value=0.0, value=3.5)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN"]
    )

st.divider()

# Prediction
if st.button("Predict", use_container_width=True):

    # One-hot encoding for ocean_proximity (same as training)
    input_data = pd.DataFrame(
        [[
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income,
            1 if ocean_proximity == "INLAND" else 0,
            1 if ocean_proximity == "NEAR BAY" else 0,
            1 if ocean_proximity == "NEAR OCEAN" else 0,
        ]],
        columns=features
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Median House Value: ${prediction:,.0f}")
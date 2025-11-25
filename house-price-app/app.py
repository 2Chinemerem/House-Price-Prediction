import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("🏠 House Price Prediction App")

# Load Models
with open(os.path.join(BASE_DIR, "regressor.pkl"), "rb") as f:
    reg_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "lasso.pkl"), "rb") as f:
    lasso_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "ridge.pkl"), "rb") as f:
    ridge_model = pickle.load(f)


# Load Scaler & Encoder
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

with open(os.path.join(BASE_DIR, "poly_feature.pkl"), "rb") as f:
    poly = pickle.load(f)



# Model Selection
model_choice = st.selectbox(
    "Select Model",
    ["Linear Regression", "LASSO", "Ridge"]
)


# User Inputs
median_age = st.number_input("Median Age (Median age of the house)", 0.0, 100.0, 20.0)
total_rooms = st.number_input("Total Rooms (Total number of rooms within a block)", 0.0, value=1000.0)
total_bedrooms = st.number_input("Total Bedrooms (Total number of bedrooms within a block)", 0.0, value=200.0)
population = st.number_input("Population (Total number of people residing within a block)", 0.0, value=800.0)
households = st.number_input("Households (Total number of households living in a block)", 0.0, value=300.0)
median_income = st.number_input("Median Income of residents", 0.0, value=3.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity (Location of the house w.r.t ocean/sea)",
    encoder.classes_
)


# Prediction Button
if st.button("Predict Price"):

    # Encode categorical variable
    ocean_encoded = encoder.transform([ocean_proximity])[0]

    # Prepare raw input
    input_data = np.array([[
        median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        ocean_encoded
    ]])

  
    input_data= poly.transform(input_data)
    scaled_input = scaler.transform(input_data)

    if model_choice == "Linear Regression":
        model = reg_model

    elif model_choice == "LASSO":
        model = lasso_model

    else:
        model = ridge_model

    # Predict
    prediction = model.predict(scaled_input)[0]

    st.success(f"💰Predicted House Price: ${prediction:,.2f}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🎯 Predict Hearing Thresholds")

# --- Select frequency and model ---
frequencies = ["500 Hz", "1000 Hz", "2000 Hz", "4000 Hz"]
models_available = ["LR", "RF", "SVM", "DT", "KNN"]

selected_freq = st.selectbox("Select Frequency", frequencies)
selected_model = st.selectbox("Select Model", models_available)

# Load the model and scaler if available
model_path = f"saved_models/{selected_freq}_{selected_model}.pkl"
scaler_path = f"saved_models/{selected_freq}_{selected_model}_scaler.pkl"

if not os.path.exists(model_path):
    st.warning("Model not found! Please train and save the model first on the main page.")
else:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # --- Input features ---
    st.subheader("Enter Patient Features")
    otoscope = st.selectbox("Otoscope", ["Normal", "Abnormal"])
    tympanometry = st.selectbox("Tympanometry", ["Type A", "Type B", "Type C"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    assr_500 = st.number_input("ASSR-500 Hz", min_value=0, max_value=120, value=20)
    assr_1000 = st.number_input("ASSR-1000 Hz", min_value=0, max_value=120, value=20)
    assr_2000 = st.number_input("ASSR-2000 Hz", min_value=0, max_value=120, value=20)
    assr_4000 = st.number_input("ASSR-4000 Hz", min_value=0, max_value=120, value=20)

    # Build dataframe
    input_dict = {
        "OTOSCOPE": [otoscope],
        "TYMPANOMETRY": [tympanometry],
        "GENDER": [gender],
        "AGE": [age],
        "ASSR-500 Hz": [assr_500],
        "ASSR-1000 Hz": [assr_1000],
        "ASSR-2000 Hz": [assr_2000],
        "ASSR-4000 Hz": [assr_4000],
    }
    input_df = pd.DataFrame(input_dict)

    # One-hot encode categorical features (like training)
    input_df = pd.get_dummies(input_df, drop_first=True)
    # Keep only numeric columns
    input_df = input_df.select_dtypes(include=[np.number])

    # Align columns with training data
    # Some columns might be missing if category wasn't present in input
    if scaler:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values

    # --- Prediction ---
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted PTA ({selected_freq}) Threshold: {prediction:.1f} dB")

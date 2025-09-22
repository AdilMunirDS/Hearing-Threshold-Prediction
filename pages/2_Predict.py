import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("Predict Hearing Thresholds")

frequencies = ["500 Hz", "1000 Hz", "2000 Hz", "4000 Hz"]
models_available = ["LR", "RF", "SVM", "DT", "KNN"]

selected_freq = st.selectbox("Select Frequency", frequencies)
selected_model = st.selectbox("Select Model", models_available)

model_path = f"saved_models/{selected_freq}_{selected_model}.pkl"

if not os.path.exists(model_path):
    st.warning("⚠️ Model not found! Please train and save the model first on the main page.")
else:
    # Load bundle: (model, scaler, feature_columns)
    model, scaler, feature_columns = joblib.load(model_path)

    st.subheader("Enter Patient Features")
    otoscope = st.selectbox("Otoscope", ["Normal", "Abnormal"])
    tympanometry = st.selectbox("Tympanometry", ["Type A", "Type B", "Type C"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    assr_500 = st.number_input("ASSR-500 Hz", min_value=0, max_value=120, value=20)
    assr_1000 = st.number_input("ASSR-1000 Hz", min_value=0, max_value=120, value=20)
    assr_2000 = st.number_input("ASSR-2000 Hz", min_value=0, max_value=120, value=20)
    assr_4000 = st.number_input("ASSR-4000 Hz", min_value=0, max_value=120, value=20)

    # Build input dataframe
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

  
    input_df = pd.get_dummies(input_df, drop_first=True)

    
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

 
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values


    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted PTA ({selected_freq}) Threshold: {prediction:.1f} dB")

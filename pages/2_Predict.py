import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🔮 Predict Hearing Thresholds")

# Frequency & Models
frequencies = ["500 Hz", "1000 Hz", "2000 Hz", "4000 Hz"]
models = ["LR", "RF", "SVM", "DT", "KNN"]

selected_freq = st.selectbox("🎚 Select Frequency", frequencies)
selected_model = st.selectbox("🤖 Select Model", models)

# User inputs
otoscope = st.selectbox("OTOSCOPE", ["Normal", "Abnormal"], index=0)
tymp = st.selectbox("TYMPANOMETRY", ["Type A", "Type B", "Type C"], index=0)
gender = st.selectbox("Gender", ["Male", "Female"], index=0)
age = st.number_input("Age", min_value=0, max_value=100, value=30)

assr_500 = st.number_input("ASSR-500 Hz", value=20)
assr_1000 = st.number_input("ASSR-1000 Hz", value=20)
assr_2000 = st.number_input("ASSR-2000 Hz", value=20)
assr_4000 = st.number_input("ASSR-4000 Hz", value=20)

# Build input row
input_dict = {
    "AGE": age,
    "ASSR-500 Hz": assr_500,
    "ASSR-1000 Hz": assr_1000,
    "ASSR-2000 Hz": assr_2000,
    "ASSR-4000 Hz": assr_4000,
    f"OTOSCOPE_{otoscope}": 1,
    f"TYMPANOMETRY_{tymp}": 1,
    f"GENDER_{gender}": 1,
}
X_input = pd.DataFrame([input_dict]).fillna(0)

# Load saved model
model_path = f"saved_models/{selected_freq}_{selected_model}.pkl"
scaler_path = f"saved_models/{selected_freq}_{selected_model}_scaler.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_input = scaler.transform(X_input)

    if st.button("🔍 Predict"):
        pred = model.predict(X_input)
        st.success(f"🎯 Predicted {selected_freq} PTA: {pred[0]:.2f} dB HL")
else:
    st.error("⚠️ Model not trained/saved yet. Please train first on Home page.")

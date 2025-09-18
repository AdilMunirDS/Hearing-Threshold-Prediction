import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🎯 Predict Hearing Threshold (PTA)")

# Frequency mapping
frequencies = {
    "500 Hz": "PTA-500 Hz",
    "1000 Hz": "PTA-1000 Hz",
    "2000 Hz": "PTA-2000 Hz",
    "4000 Hz": "PTA-4000 Hz",
}

# Check saved models
if not os.path.exists("saved_models"):
    st.warning("⚠️ No saved models found. Please go to the **Home** page to train & save models first.")
else:
    available_models = [f for f in os.listdir("saved_models") if f.endswith(".pkl")]
    if not available_models:
        st.warning("⚠️ No saved models available yet. Train & save them from the **Home** page.")
    else:
        # User selects frequency
        selected_freq = st.selectbox("🎚 Select Frequency", list(frequencies.keys()))

        # Detect models available for that frequency
        models_for_freq = [m for m in available_models if selected_freq.replace(" ", "_") in m or selected_freq in m]
        model_options = []
        for m in models_for_freq:
            if "LR" in m: model_options.append("Linear Regression")
            elif "RF" in m: model_options.append("Random Forest")
            elif "SVM" in m: model_options.append("SVM")
            elif "DT" in m: model_options.append("Decision Tree")
            elif "KNN" in m: model_options.append("KNN")

        if not model_options:
            st.warning(f"⚠️ No models found for {selected_freq}. Train them first.")
        else:
            selected_model = st.selectbox("🤖 Select Model", model_options)

            # --- User Inputs ---
            st.subheader("📥 Enter Input Data")
            otoscope = st.selectbox("Otoscopy", ["Normal", "Abnormal"], index=0)
            tymp = st.selectbox("Tympanometry", ["Type A", "Type B", "Type C"], index=0)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

            assr_500 = st.number_input("ASSR 500 Hz", value=20)
            assr_1000 = st.number_input("ASSR 1000 Hz", value=20)
            assr_2000 = st.number_input("ASSR 2000 Hz", value=20)
            assr_4000 = st.number_input("ASSR 4000 Hz", value=20)

            if st.button("🔮 Predict PTA"):
                # Prepare input dataframe
                input_dict = {
                    "AGE": age,
                    "ASSR-500 Hz": assr_500,
                    "ASSR-1000 Hz": assr_1000,
                    "ASSR-2000 Hz": assr_2000,
                    "ASSR-4000 Hz": assr_4000,
                    "OTOSCOPE": otoscope,
                    "TYMPANOMETRY": tymp,
                    "GENDER": gender,
                }
                input_df = pd.DataFrame([input_dict])
                input_df = pd.get_dummies(input_df, drop_first=True)

                # Align columns with training time
                model_key = selected_model.split()[0]
                model_filename = f"saved_models/{selected_freq}_{model_key}.pkl"
                scaler_filename = f"saved_models/{selected_freq}_{model_key}_scaler.pkl"

                # Load model
                if not os.path.exists(model_filename):
                    st.error("Model file not found. Please retrain.")
                else:
                    model = joblib.load(model_filename)

                    # Load scaler if available
                    if os.path.exists(scaler_filename):
                        scaler = joblib.load(scaler_filename)
                        input_scaled = scaler.transform(input_df)
                        prediction = model.predict(input_scaled)[0]
                    else:
                        prediction = model.predict(input_df)[0]

                    st.success(f"✅ Predicted PTA at {selected_freq}: **{prediction:.2f} dB HL**")

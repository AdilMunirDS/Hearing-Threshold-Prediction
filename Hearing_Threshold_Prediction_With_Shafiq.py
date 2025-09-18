import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------
# App title
# ---------------------------
st.title("Hearing Threshold Prediction App")

# ---------------------------
# File uploader
# ---------------------------
uploaded_file = st.file_uploader("Upload your Excel/CSV dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Select Features and Target
    # ---------------------------
    st.subheader("Model Training")

    features = st.multiselect("Select Feature Columns", df.columns.tolist())
    target = st.selectbox("Select Target Column", df.columns.tolist())

    if features and target:
        X = df[features]
        y = df[target]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---------------------------
        # Model Selection
        # ---------------------------
        model_choice = st.selectbox(
            "Choose a model",
            ["Linear Regression", "Random Forest", "SVM", "Decision Tree", "KNN"]
        )

        if st.button("Train Model"):
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor(random_state=42)
            elif model_choice == "SVM":
                model = SVR()
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
            elif model_choice == "KNN":
                model = KNeighborsRegressor()

            # Train
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)

            # Accuracy within ±10 dB
            within_10 = np.mean(np.abs(y_pred - y_test) <= 10) * 100
            st.success(f"Model trained successfully! Accuracy within ±10 dB: {within_10:.2f}%")

            # Save model and scaler
            joblib.dump(model, "trained_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            st.info("Model and scaler saved as 'trained_model.pkl' & 'scaler.pkl'")

# ---------------------------
# Prediction Section
# ---------------------------
st.subheader("Make a Prediction with Inputs")

# Input fields
assr_500 = st.text_input("ASSR at 500 Hz", "")
assr_1000 = st.text_input("ASSR at 1000 Hz", "")
assr_2000 = st.text_input("ASSR at 2000 Hz", "")
assr_4000 = st.text_input("ASSR at 4000 Hz", "")
age = st.text_input("Age", "")

gender = st.selectbox("Gender", ["Male", "Female"])
otoscopy = st.selectbox("Otoscopy", ["Normal", "Wax", "Perforation", "Other"])

if st.button("Predict PTA"):
    if os.path.exists("trained_model.pkl") and os.path.exists("scaler.pkl"):
        model = joblib.load("trained_model.pkl")
        scaler = joblib.load("scaler.pkl")

        try:
            # Build input
            input_data = pd.DataFrame([[
                float(assr_500), float(assr_1000), float(assr_2000), float(assr_4000), float(age),
                1 if gender == "Male" else 0,
                {"Normal":0, "Wax":1, "Perforation":2, "Other":3}[otoscopy]
            ]])

            # Scale features
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)
            st.success(f"Predicted PTA Threshold: {prediction[0]:.2f} dB HL")

        except ValueError:
            st.error("Please enter valid numeric values for ASSR and Age fields.")
    else:
        st.warning("No trained model found. Please train a model first.")

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
from sklearn.multioutput import MultiOutputRegressor

# -----------------------------
# Helper functions
# -----------------------------
def get_model(model_name):
    if model_name == "Linear Regression":
        return LinearRegression()
    elif model_name == "Random Forest":
        return RandomForestRegressor(random_state=42)
    elif model_name == "SVM":
        return MultiOutputRegressor(SVR())  # SVR doesn't support multi-output directly
    elif model_name == "Decision Tree":
        return DecisionTreeRegressor(random_state=42)
    elif model_name == "KNN":
        return KNeighborsRegressor()
    else:
        return None

def save_model(model, scaler, features, targets, filename="trained_model.pkl"):
    joblib.dump({"model": model, "scaler": scaler, "features": features, "targets": targets}, filename)

def load_model(filename="trained_model.pkl"):
    return joblib.load(filename)

def acc_10_db(y_true, y_pred):
    """Calculate ±10 dB accuracy for each output column."""
    return np.mean(np.abs(y_true - y_pred) <= 10, axis=0) * 100

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Multi-Frequency PTA Prediction from ASSR")

menu = ["Upload & Train", "Predict", "Load Saved Model"]
choice = st.sidebar.radio("Menu", menu)

if choice == "Upload & Train":
    st.subheader("Upload Dataset & Train Model")

    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Dataset Preview:", df.head())

        # Select input (X) and target (y)
        features = st.multiselect("Select input features (ASSR)", df.columns.tolist())
        targets = st.multiselect("Select target columns (PTA)", df.columns.tolist())

        if features and targets:
            X = df[features].values
            y = df[targets].values

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Choose model
            model_name = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "SVM", "Decision Tree", "KNN"])
            model = get_model(model_name)

            if st.button("Train Model"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # ±10 dB accuracy
                acc = acc_10_db(y_test, y_pred)

                st.success(f"Model Trained: {model_name}")
                for i, target in enumerate(targets):
                    st.write(f"{target} → ±10 dB Accuracy: {acc[i]:.2f}%")

                if st.button("Save Trained Model"):
                    save_model(model, scaler, features, targets, "trained_model.pkl")
                    st.success("Model saved successfully as trained_model.pkl")

elif choice == "Predict":
    st.subheader("Predict PTA from ASSR")

    if os.path.exists("trained_model.pkl"):
        model_data = load_model("trained_model.pkl")
        model = model_data["model"]
        scaler = model_data["scaler"]
        features = model_data["features"]
        targets = model_data["targets"]

        st.write(f"Expected ASSR input features: {features}")

        # Input ASSR values
        input_values = st.text_input(f"Enter ASSR values for {features} (comma separated)")
        if input_values:
            try:
                input_list = [float(x.strip()) for x in input_values.split(",")]
                if len(input_list) != len(features):
                    st.error(f"Expected {len(features)} values, got {len(input_list)}")
                else:
                    input_array = np.array(input_list).reshape(1, -1)
                    input_scaled = scaler.transform(input_array)

                    prediction = model.predict(input_scaled)[0]

                    st.success("Predicted PTA values:")
                    for i, target in enumerate(targets):
                        st.write(f"{target}: {prediction[i]:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("No trained model found. Please train and save a model first.")

elif choice == "Load Saved Model":
    st.subheader("Load a Previously Saved Model")
    model_file = st.file_uploader("Upload a trained model (.pkl)", type=["pkl"])
    if model_file is not None:
        model_data = joblib.load(model_file)
        st.success("Model loaded successfully! You can now use it in the Predict tab.")
        save_model(model_data["model"], model_data["scaler"], model_data["features"], model_data["targets"], "trained_model.pkl")

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

st.title("📊 Train & Save Models for Hearing Threshold Prediction")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    df = load_data(uploaded_file)
    st.success("✅ File loaded successfully!")
    st.write(df.head())

    # Frequency mapping
    frequencies = {
        "500 Hz": "PTA-500 Hz",
        "1000 Hz": "PTA-1000 Hz",
        "2000 Hz": "PTA-2000 Hz",
        "4000 Hz": "PTA-4000 Hz",
    }

    # Features
    features = [
        "OTOSCOPE", "TYMPANOMETRY", "GENDER", "AGE",
        "ASSR-500 Hz", "ASSR-1000 Hz", "ASSR-2000 Hz", "ASSR-4000 Hz"
    ]

    selected_freq = st.selectbox("🎚 Select Frequency to Predict", list(frequencies.keys()))
    target_col = frequencies[selected_freq]

    data = df[features + [target_col]].dropna()
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def acc_10_db(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred) <= 10) * 100

    # --- Linear Regression ---
    st.subheader("Linear Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("💾 Save LR Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(lr, f"saved_models/{selected_freq}_LR.pkl")
        joblib.dump(scaler, f"saved_models/{selected_freq}_LR_scaler.pkl")
        st.success("LR + Scaler saved!")

    # --- Random Forest ---
    st.subheader("Random Forest")
    n_estimators = st.slider("Number of Trees (n_estimators)", 10, 300, 100, step=10)
    max_depth = st.slider("Max Depth", 1, 50, 10)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("💾 Save RF Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(rf, f"saved_models/{selected_freq}_RF.pkl")
        st.success("RF saved!")

    # --- SVM ---
    st.subheader("SVM")
    C = st.number_input("C (Regularization)", 0.1, 10.0, 1.0, step=0.1)
    epsilon = st.number_input("Epsilon (Margin)", 0.0, 2.0, 0.1, step=0.1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm = SVR(C=C, epsilon=epsilon)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("💾 Save SVM Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(svm, f"saved_models/{selected_freq}_SVM.pkl")
        joblib.dump(scaler, f"saved_models/{selected_freq}_SVM_scaler.pkl")
        st.success("SVM + Scaler saved!")

    # --- Decision Tree ---
    st.subheader("Decision Tree")
    max_depth = st.slider("Max Depth (Decision Tree)", 1, 50, 5)
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("💾 Save DT Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(dt, f"saved_models/{selected_freq}_DT.pkl")
        st.success("DT saved!")

    # --- KNN ---
    st.subheader("KNN")
    n_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 5)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("💾 Save KNN Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(knn, f"saved_models/{selected_freq}_KNN.pkl")
        joblib.dump(scaler, f"saved_models/{selected_freq}_KNN_scaler.pkl")
        st.success("KNN + Scaler saved!")

else:
    st.info("📂 Please upload an Excel file to proceed.")

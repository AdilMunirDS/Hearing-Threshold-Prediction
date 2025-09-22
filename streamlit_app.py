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

st.title("Train & Save Models for Hearing Threshold Prediction")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    df = load_data(uploaded_file)
    st.success("File loaded successfully!")
    st.write(df.head())

    # Frequencies mapping
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

    selected_freq = st.selectbox("Select Frequency to Predict", list(frequencies.keys()))
    target_col = frequencies[selected_freq]

    data = df[features + [target_col]].dropna()

    # Save categories for selectboxes later
    categorical_columns = ["OTOSCOPE", "TYMPANOMETRY", "GENDER"]
    categories_dict = {col: data[col].unique().tolist() for col in categorical_columns}

    # One-hot encode categorical features
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def acc_10_db(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred) <= 10) * 100

    # Sidebar hyperparameters
    st.sidebar.header("Hyperparameters")
    rf_max_depth = st.sidebar.slider("Random Forest max_depth", 2, 20, 10)
    rf_n_estimators = st.sidebar.slider("Random Forest n_estimators", 10, 200, 100)
    dt_max_depth = st.sidebar.slider("Decision Tree max_depth", 2, 20, 5)
    knn_neighbors = st.sidebar.slider("KNN n_neighbors", 1, 20, 5)
    svm_c = st.sidebar.number_input("SVM C", 0.1, 10.0, 1.0)
    svm_epsilon = st.sidebar.number_input("SVM epsilon", 0.01, 1.0, 0.1)

    # ========== Linear Regression ==========
    st.subheader("Linear Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("Save LR Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump((lr, scaler, X.columns.tolist(), categories_dict),
                    f"saved_models/{selected_freq}_LR.pkl")
        st.success("LR + Scaler + Features + Categories saved!")

    # ========== Random Forest ==========
    st.subheader("Random Forest")
    rf = RandomForestRegressor(n_estimators=rf_n_estimators,
                               max_depth=rf_max_depth,
                               random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("Save RF Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump((rf, None, X.columns.tolist(), categories_dict),
                    f"saved_models/{selected_freq}_RF.pkl")
        st.success("RF + Features + Categories saved!")

    # ========== SVM ==========
    st.subheader("SVM")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm = SVR(C=svm_c, epsilon=svm_epsilon)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("Save SVM Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump((svm, scaler, X.columns.tolist(), categories_dict),
                    f"saved_models/{selected_freq}_SVM.pkl")
        st.success("SVM + Scaler + Features + Categories saved!")

    # ========== Decision Tree ==========
    st.subheader("Decision Tree")
    dt = DecisionTreeRegressor(max_depth=dt_max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("Save DT Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump((dt, None, X.columns.tolist(), categories_dict),
                    f"saved_models/{selected_freq}_DT.pkl")
        st.success("DT + Features + Categories saved!")

    # ========== KNN ==========
    st.subheader("KNN")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsRegressor(n_neighbors=knn_neighbors)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")
    if st.button("Save KNN Model"):
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump((knn, scaler, X.columns.tolist(), categories_dict),
                    f"saved_models/{selected_freq}_KNN.pkl")
        st.success("KNN + Scaler + Features + Categories saved!")

else:
    st.info("Please upload an Excel file to proceed.")

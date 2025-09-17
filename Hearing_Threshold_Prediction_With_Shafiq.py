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


st.title("Hearing Threshold Prediction")


uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)
    
    df = load_data(uploaded_file)
    st.success("File loaded successfully!")
    st.write(df.head())

  
    frequencies = {
        "500 Hz": "PTA-500 Hz",
        "1000 Hz": "PTA-1000 Hz",
        "2000 Hz": "PTA-2000 Hz",
        "4000 Hz": "PTA-4000 Hz",
    }


    features = [
        "OTOSCOPE", "TYMPANOMETRY", "GENDER", "AGE",
        "ASSR-500 Hz", "ASSR-1000 Hz", "ASSR-2000 Hz", "ASSR-4000 Hz"
    ]

    selected_freq = st.selectbox("Select Frequency to Predict", list(frequencies.keys()))
    target_col = frequencies[selected_freq]

   
    data = df[features + [target_col]].dropna()
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    def acc_10_db(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred) <= 10) * 100

 
    with col1:
        st.subheader("Linear Regression")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")

        if st.button("Save LR Model"):
            os.makedirs("saved_models", exist_ok=True)
            joblib.dump(model, f"saved_models/model_{selected_freq}_LR.pkl")
            joblib.dump(scaler, f"saved_models/model_{selected_freq}_LR_scaler.pkl")
            st.success("Saved LR + scaler")


    with col2:
        st.subheader("Random Forest")
        n_estimators = st.slider("n_estimators", 10, 200, 100, step=10, key="rf_ne")
        max_depth = st.slider("max_depth", 1, 30, 10, key="rf_md")
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")

        if st.button("Save RF Model"):
            os.makedirs("saved_models", exist_ok=True)
            joblib.dump(model, f"saved_models/model_{selected_freq}_RF.pkl")
            st.success("Saved RF")

  
    with col3:
        st.subheader("SVM")
        C = st.number_input("C", 0.1, 10.0, 1.0, step=0.1, key="svm_c")
        epsilon = st.number_input("epsilon", 0.0, 2.0, 0.1, step=0.1, key="svm_eps")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = SVR(C=C, epsilon=epsilon)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")

        if st.button("Save SVM Model"):
            os.makedirs("saved_models", exist_ok=True)
            joblib.dump(model, f"saved_models/model_{selected_freq}_SVM.pkl")
            joblib.dump(scaler, f"saved_models/model_{selected_freq}_SVM_scaler.pkl")
            st.success("Saved SVM + scaler")

    
    with col4:
        st.subheader("Decision Tree")
        max_depth = st.slider("max_depth", 1, 30, 5, key="dt_md")
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")

        if st.button("Save DT Model"):
            os.makedirs("saved_models", exist_ok=True)
            joblib.dump(model, f"saved_models/model_{selected_freq}_DT.pkl")
            st.success("Saved DT")

   
    with col5:
        st.subheader("KNN")
        n_neighbors = st.slider("n_neighbors", 1, 20, 5, key="knn_k")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.write(f"±10 dB Accuracy: {acc_10_db(y_test, y_pred):.2f}%")

        if st.button("Save KNN Model"):
            os.makedirs("saved_models", exist_ok=True)
            joblib.dump(model, f"saved_models/model_{selected_freq}_KNN.pkl")
            joblib.dump(scaler, f"saved_models/model_{selected_freq}_KNN_scaler.pkl")
            st.success("Saved KNN + scaler")

else:
    st.info("Please upload an Excel file to proceed.")

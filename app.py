import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


MODELS_DIR = "models"

@st.cache_resource
def load_artifacts():
    model = load_model(os.path.join(MODELS_DIR, 'lstm_model.h5'))
    feature_scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    return model, feature_scaler, target_scaler, feature_columns

st.title("Cholera Outbreak Prediction")

model, feature_scaler, target_scaler, feature_columns = load_artifacts()

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        X = df[feature_columns].fillna(0)
        X_scaled = feature_scaler.transform(X)
        X_r = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        preds_scaled = model.predict(X_r)
        preds = target_scaler.inverse_transform(preds_scaled).flatten()

        results_df = pd.DataFrame({'Predicted_Cholera_Cases': preds}, index=df.index)
        st.line_chart(results_df)
        st.dataframe(results_df)

        csv = results_df.to_csv().encode('utf-8')
        st.download_button("Download predictions", data=csv, file_name='predictions.csv', mime='text/csv')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODELS_DIR = "models"

@st.cache_resource
def load_artifacts():
    model = load_model(os.path.join(MODELS_DIR, 'final_lstm_model.h5'))
    feature_scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    return model, feature_scaler, target_scaler, feature_columns

st.title("Cholera Outbreak Prediction")

model, feature_scaler, target_scaler, feature_columns = load_artifacts()

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        X = df[feature_columns].fillna(0)
        X_scaled = feature_scaler.transform(X)
        X_r = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        preds_scaled = model.predict(X_r)
        preds = target_scaler.inverse_transform(preds_scaled).flatten()

        results_df = pd.DataFrame({'Predicted_Cholera_Cases': preds}, index=df.index)
        st.line_chart(results_df)
        st.dataframe(results_df)

        csv = results_df.to_csv().encode('utf-8')
        st.download_button("Download predictions", data=csv, file_name='predictions.csv', mime='text/csv')

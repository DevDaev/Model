import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
    if 'Cholera_Cases' not in df.columns:
        raise ValueError("CSV must contain a 'Cholera_Cases' column.")
    return df

def preprocess(df):
    features = df.drop(columns=['Cholera_Cases'])
    target = df['Cholera_Cases']
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    target = target.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return features, target

def build_model(n_features, lstm_units=50, learning_rate=0.0005):
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=(1, n_features)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def main(args):
    print("Attempting to create directory....")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Directory '{args.output_dir}' created successfully.")

    df = load_data(args.data)
    features, target = preprocess(df)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    X_train_r = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_r = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = build_model(X_train.shape[1], args.lstm_units, args.learning_rate)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(os.path.join(args.output_dir, 'final_lstm_model.h5'), save_best_only=True)
    ]

    model.fit(X_train_r, y_train_scaled, epochs=args.epochs, batch_size=args.batch_size,
              validation_split=0.2, callbacks=callbacks, verbose=1)
    
    print("Saving models...")
    model.save(os.path.join(args.output_dir, 'final_lstm_model.h5')
    joblib.dump(feature_scaler, os.path.join(args.output_dir, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(args.output_dir, 'target_scaler.pkl'))
    joblib.dump(list(features.columns), os.path.join(args.output_dir, 'feature_columns.pkl'))
    print("Models saved successfully.")
    
    preds_scaled = model.predict(X_test_r)
    preds = target_scaler.inverse_transform(preds_scaled).flatten()
    y_test_unscaled = y_test.values.flatten()

    mse = mean_squared_error(y_test_unscaled, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, preds)
    print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/lstm_cholera_forecasting_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lstm_units', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    args = parser.parse_args()




import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv1D, GlobalAveragePooling1D

def build_lstm(seq_len, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        LSTM(32, return_sequences=False),
        Dense(1)
    ], name='LSTM_Model')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru(seq_len, n_features):
    # no dropout — GRU has fewer parameters than LSTM so overfitting risk is lower
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_len, n_features)),
        GRU(32, return_sequences=False),
        Dense(1)
    ], name='GRU_Model')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_tcn(seq_len, n_features):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu',
               padding='causal', input_shape=(seq_len, n_features)),
        Conv1D(64, kernel_size=3, activation='relu',
               padding='causal', dilation_rate=2),
        Conv1D(32, kernel_size=3, activation='relu',
               padding='causal', dilation_rate=4),
        GlobalAveragePooling1D(),
        Dense(1)
    ], name='TCN_Model')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def compute_metrics(y_true_sc, y_pred_sc, scaler, target_idx):
    """
    Inverse-transform scaled predictions back to µg/m³ and compute
    MAE, MSE, RMSE, MAPE on the original NO2 scale.
    Dummy arrays are needed because the scaler was fitted on all 8 columns.
    """
    n = len(y_true_sc)
    dummy_true = np.zeros((n, scaler.n_features_in_))
    dummy_pred = np.zeros((n, scaler.n_features_in_))
    
    dummy_true[:, target_idx] = y_true_sc
    dummy_pred[:, target_idx] = y_pred_sc

    y_true = scaler.inverse_transform(dummy_true)[:, target_idx]
    y_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4)
    }
    
    return metrics, y_true, y_pred

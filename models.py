import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Conv1D, Dense, GlobalAveragePooling1D, GRU, LSTM
from tensorflow.keras.models import Sequential


def build_lstm(seq_len, n_features):
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
            LSTM(32),
            Dense(1),
        ],
        name="LSTM_Model",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_gru(seq_len, n_features):
    model = Sequential(
        [
            GRU(64, return_sequences=True, input_shape=(seq_len, n_features)),
            GRU(32),
            Dense(1),
        ],
        name="GRU_Model",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_tcn(seq_len, n_features):
    model = Sequential(
        [
            Conv1D(64, 3, activation="relu", padding="causal", input_shape=(seq_len, n_features)),
            Conv1D(64, 3, activation="relu", padding="causal", dilation_rate=2),
            Conv1D(32, 3, activation="relu", padding="causal", dilation_rate=4),
            GlobalAveragePooling1D(),
            Dense(1),
        ],
        name="TCN_Model",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def inverse_target(values, scaler, target_idx):
    data = np.zeros((len(values), scaler.n_features_in_))
    data[:, target_idx] = values
    return scaler.inverse_transform(data)[:, target_idx]


def compute_metrics(y_true_sc, y_pred_sc, scaler, target_idx):
    y_true = inverse_target(y_true_sc, scaler, target_idx)
    y_pred = inverse_target(y_pred_sc, scaler, target_idx)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true[y_true > 0] - y_pred[y_true > 0]) / y_true[y_true > 0])) * 100

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
    }, y_true, y_pred

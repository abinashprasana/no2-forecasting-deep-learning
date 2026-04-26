# NO2 Concentration Forecasting Using Deep Learning: London Marylebone Road (2025)

This repository contains a deep learning project that forecasts urban air pollution (NO2 levels) at London Marylebone Road using a 24-hour sliding window sequence approach.

## 📊 Dataset Details
*   **Name/Source:** UK DEFRA AURN Dataset (London Marylebone Road)
*   **Size:** 7,843 hours (rows) spanning from 2025-01-01 to 2026-01-01
*   **Features:** 8 pollutant features (CO, PM10, NO, NO2, NOx, O3, PM2.5, SO2) with NO2 as the target variable

## ✂️ Data Splitting & Windowing
*   **Split Ratios:** 70% Train / 15% Validation / 15% Test (Strict chronological split to prevent data leakage)
*   **Sliding Window Size:** 24 hours (Sequence length of 24 to predict the 25th hour's NO2 concentration)

## 🧹 Preprocessing Steps
1.  **Datetime Alignment:** Resolved DEFRA's specific "24:00" hour format by shifting dates forward 1 day and setting to "00:00".
2.  **Artefact Removal:** Replaced negative instrument reading artefacts with `NaN`.
3.  **Outlier Handling:** Clipped extreme upper-end outliers at the 99th percentile.
4.  **Imputation:** Applied linear interpolation (limit of 6 hours in both directions) to fill short data gaps, then dropped remaining NaN values.
5.  **Scaling:** Standardized all features to a `[0, 1]` range using a `MinMaxScaler` fitted **exclusively** on the training set.

## 🧠 Model Architectures
**1. LSTM (Long Short-Term Memory)**
*   **Layer 1:** LSTM (64 units, `return_sequences=True`)
*   **Layer 2:** LSTM (32 units, `return_sequences=False`)
*   **Output Layer:** Dense (1 unit)
*   **Key Settings:** Optimizer='adam', Loss='mse'

**2. GRU (Gated Recurrent Unit)**
*   **Layer 1:** GRU (64 units, `return_sequences=True`)
*   **Layer 2:** GRU (32 units, `return_sequences=False`)
*   **Output Layer:** Dense (1 unit)
*   **Key Settings:** Optimizer='adam', Loss='mse' (No dropout used as GRU naturally has fewer parameters, lowering overfitting risk)

**3. TCN (Temporal Convolutional Network)**
*   **Layer 1:** Conv1D (64 filters, kernel=3, relu, padding='causal', dilation=1)
*   **Layer 2:** Conv1D (64 filters, kernel=3, relu, padding='causal', dilation_rate=2)
*   **Layer 3:** Conv1D (32 filters, kernel=3, relu, padding='causal', dilation_rate=4)
*   **Pooling:** GlobalAveragePooling1D
*   **Output Layer:** Dense (1 unit)
*   **Key Settings:** Optimizer='adam', Loss='mse'

## 🏆 Model Performance Metrics (Test Set)
| Model | MAE | MSE | RMSE | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM** | **4.5460** | **37.0458** | **6.0865** | **22.9770** |
| **GRU** | 5.9759 | 57.5909 | 7.5889 | 36.6754 |
| **TCN** | 10.8209 | 183.8128 | 13.5578 | 72.0659 |

**Best Model:** The LSTM performed the best across all metrics. It outperformed the GRU by 1.5024 RMSE (a ~20% relative improvement) and outperformed the TCN by a massive 7.4713 RMSE. 

## 🌐 Flask Web App Structure
The `app.py` script serves a responsive HTML dashboard displaying the following sections:
1.  **Exploratory Data Analysis:** Contextual text and the 4 EDA plots.
2.  **Model Training and Validation:** Explanations regarding the loss curves and the `training_loss_curves.png` image.
3.  **Forecasting Results:** Text summarizing test set performance alongside `forecast_comparison.png` and `forecast_scatter.png`.
4.  **Final Evaluation Metrics:** A dynamic HTML table parsing `results.json` and the `results_comparison.png` bar chart.

## ⚙️ Execution Commands
```bash
pip install -r requirements.txt
python train.py
python app.py
```
Open `http://127.0.0.1:5000` to view the dashboard!

## 🔍 Pipeline Simulation Trace
1.  **Raw Input:** An hour arrives with 8 uncleaned pollutant measurements.
2.  **Cleaning:** The pipeline ensures the datetime is correctly formatted, replaces any negative instrument readings with `NaN`, clips any crazy upper spikes, and interpolates if a sensor briefly failed.
3.  **Scaling:** The 8 cleaned values are squeezed into a `[0, 1]` range using the pre-fitted MinMaxScaler.
4.  **Windowing:** This newly scaled hour is appended to the 23 previous hours to create a chronological sequence block of shape `(24, 8)`.
5.  **Forward Pass:** The `(24, 8)` matrix is passed into the best model (LSTM). The LSTM layers read the temporal sequence and the Dense layer spits out a single scalar value.
6.  **Inverse Scaling:** The value is mapped to the NO2 index of a dummy array. `scaler.inverse_transform` converts it back to standard units, resulting in a human-readable prediction (µg/m³).

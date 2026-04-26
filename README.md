# <h1 align="center">🌫️ NO2 Forecasting — Urban Air Quality Prediction with Deep Learning</h1>

<p align="center">
  Three deep learning models (LSTM, GRU, TCN) trained on real UK air quality sensor data to forecast hourly NO2 concentrations in London.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
  <img alt="Flask" src="https://img.shields.io/badge/Dashboard-Flask-000000?logo=flask&logoColor=white" />
  <img alt="Dataset" src="https://img.shields.io/badge/Data-UK%20DEFRA%20AURN-0078D4" />
  <img alt="Best RMSE" src="https://img.shields.io/badge/Best%20RMSE-6.09%20µg%2Fm³-2ea44f" />
  <img alt="Status" src="https://img.shields.io/badge/Status-Completed-2ea44f" />
</p>

---

## 🔎 What This Is

This project trains and compares three deep learning architectures for one-hour-ahead NO2 forecasting using real hourly sensor data from the **London Marylebone Road monitoring station**, one of the most studied urban air quality sites in the UK.

The best model (LSTM) achieves an **RMSE of 6.09 µg/m³** on the held-out test set. Results are visualised through a clean **Flask web dashboard** showing EDA plots, training curves, forecast overlays, and a model comparison table.

---

## 🎥 Demo Video

<!-- Record your screen running train.py and then the Flask dashboard, then upload here -->

https://github.com/user-attachments/assets/YOUR-VIDEO-ID-HERE

---

## 🏆 Model Performance (Test Set)

| Model | MAE | MSE | RMSE | MAPE (%) |
|---|---|---|---|---|
| **LSTM** ⭐ | **4.546** | **37.05** | **6.087** | **22.98** |
| GRU | 5.976 | 57.59 | 7.589 | 36.68 |
| TCN | 10.821 | 183.81 | 13.558 | 72.07 |

The **LSTM outperformed the GRU by ~20% on RMSE** and outperformed the TCN by a significant margin across all four metrics.

All models were trained on the same dataset under identical conditions. Metrics are computed on the original µg/m³ scale after inverse-transforming scaled predictions.

---

## 🧠 How It Works

```
Raw DEFRA CSV → Cleaning → Scaling → Sliding Window (24h)
                                              ↓
                              LSTM / GRU / TCN Training
                                              ↓
                         Inverse Transform → µg/m³ Predictions
                                              ↓
                              Flask Dashboard → Results
```

**Pipeline trace for a single prediction:**
1. Raw hourly readings arrive with 8 pollutant measurements
2. Negative instrument artefacts replaced, outliers clipped at 99th percentile
3. Short gaps filled with linear interpolation (max 6 hours)
4. All 8 features scaled to [0, 1] using MinMaxScaler fitted on training set only
5. Current hour appended to previous 23 hours → sequence shape `(24, 8)`
6. LSTM processes the sequence → outputs a single scaled scalar
7. Inverse transform converts back to µg/m³ (e.g. `0.23 → 45.2 µg/m³`)

---

## 📊 Dataset

- **Source:** UK DEFRA Automatic Urban and Rural Network (AURN)
- **Station:** London Marylebone Road (site ID: MY1) — a six-lane urban road with ~80,000 vehicles/day
- **Period:** Full calendar year 2025 (January → January)
- **Size:** 7,843 hourly rows
- **Target variable:** NO2 concentration (µg/m³)
- **Input features:** CO, PM10, NO, NO2, NOx, O3, PM2.5, SO2

**Split:** 70% Train / 15% Validation / 15% Test — strict chronological order, no shuffling

---

## 🧹 Preprocessing

- **Datetime fix:** DEFRA's `24:00` end-of-day format shifted to `00:00` next day
- **Artefact removal:** Negative instrument readings replaced with NaN
- **Outlier handling:** Values above the 99th percentile clipped per column
- **Imputation:** Linear interpolation for gaps up to 6 consecutive hours
- **Scaling:** MinMaxScaler fitted on training set only — applied separately to validation and test

---

## 🧩 Model Architectures

**LSTM (Best Performer)**
```
LSTM(64, return_sequences=True)
LSTM(32, return_sequences=False)
Dense(1)
Optimizer: Adam | Loss: MSE | Early Stopping: patience=7
```

**GRU**
```
GRU(64, return_sequences=True)
GRU(32, return_sequences=False)
Dense(1)
Optimizer: Adam | Loss: MSE | Early Stopping: patience=7
```

**TCN (Temporal Convolutional Network)**
```
Conv1D(64, kernel=3, dilation=1, causal padding)
Conv1D(64, kernel=3, dilation=2, causal padding)
Conv1D(32, kernel=3, dilation=4, causal padding)
GlobalAveragePooling1D()
Dense(1)
Optimizer: Adam | Loss: MSE | Early Stopping: patience=7
```

---

## 📈 Visualisations

**EDA:**
- All 8 pollutant time series across the full year
- NO2 daily cycle — March 2025 closeup
- Monthly average NO2 — seasonal pattern
- Feature correlation heatmap

**Evaluation:**
- Training vs validation loss curves for all three models
- Forecast overlay — actual vs predicted for first 200 test hours
- Predicted vs actual scatter plots with perfect-fit reference line
- 2×2 metric comparison bar charts (MAE, MSE, RMSE, MAPE)

---

## 🌐 Flask Dashboard

Run the app to explore all results in your browser:

**Sections:**
- 📊 Exploratory Data Analysis
- 📉 Model Training and Validation Curves
- 🔮 Forecasting Results
- 🏆 Final Evaluation Metrics Table

---

## 🗂️ Project Structure

```text
no2-forecasting/
├── data.py              # Data loading, cleaning, split, scaling, windowing
├── models.py            # LSTM, GRU, TCN builders + compute_metrics()
├── train.py             # Full training pipeline — saves plots + results.json
├── app.py               # Flask dashboard
├── static/              # Generated PNG plots (created by train.py)
├── results.json         # Model metrics (created by train.py)
├── MY1_2025.csv         # Raw DEFRA dataset (download separately)
└── requirements.txt
```

---

## ⚙️ Setup and Usage

```bash
# 1. Clone the repository
git clone https://github.com/abinashprasana/no2-forecasting-deep-learning.git
cd no2-forecasting-deep-learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Get MY1_2025.csv from: https://uk-air.defra.gov.uk/data/data_selector
# Place it in the root project folder

# 4. Train all three models (saves plots + results.json)
python train.py

# 5. Launch the Flask dashboard
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 🧪 Limitations and Future Work

- One year of data captures a single seasonal cycle, multi-year data would strengthen generalisation
- All models underpredict peak NO2 values above ~40 µg/m³, partly due to 99th percentile clipping during preprocessing
- Adding temporal features (hour-of-day, day-of-week) could improve performance, especially for the TCN
- A deeper TCN with wider dilation rates (1, 2, 4, 8, 16) would likely close the gap with the recurrent models
- The dataset comes from a single monitoring station, results may not generalise to other urban environments

---

## 📌 Data Source

**UK DEFRA Automatic Urban and Rural Network (AURN)**
Available at: https://uk-air.defra.gov.uk
Open Government Licence v3.0

---

## 🙋 Author

**Abinash Prasana Selvanathan**

⭐ If you found this useful, feel free to star the repo.

# 🌫️ NO2 Forecasting Dashboard

<p align="center">
  <strong>Hourly nitrogen dioxide forecasting for London Marylebone Road using LSTM, GRU, and TCN models.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-Project-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Training-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-Dashboard-111827?style=for-the-badge&logo=flask&logoColor=white">
  <img alt="Render" src="https://img.shields.io/badge/Render-Free_Service-46E3B7?style=for-the-badge&logo=render&logoColor=111827">
</p>

<p align="center">
  <a href="https://no2-forecasting-dashboard.onrender.com/">
    <img alt="Open live dashboard" src="https://img.shields.io/badge/Open_Live_Dashboard-0EA5E9?style=for-the-badge&logo=render&logoColor=white">
  </a>
  <a href="https://github.com/abinashprasana/no2-forecasting-deep-learning">
    <img alt="View repository" src="https://img.shields.io/badge/View_Repository-181717?style=for-the-badge&logo=github&logoColor=white">
  </a>
</p>

## What This Project Is

This is a student data science project that forecasts the next hourly NO2 value from recent air quality readings. I used real UK DEFRA AURN data from the London Marylebone Road monitoring station, cleaned the dataset, trained three deep learning models, and deployed the saved results in a Flask dashboard.

The dashboard is live here:

**https://no2-forecasting-dashboard.onrender.com/**

The hosted app shows the saved plots and saved metrics. It does not train the models online.

## Dashboard Preview

<p align="center">
  <img src="static/forecast_comparison.png" alt="Forecast comparison plot" width="48%">
  <img src="static/results_comparison.png" alt="Model comparison plot" width="48%">
</p>

## Results

The saved test metrics are computed after converting the scaled predictions back to the original NO2 scale.

| Model | MAE | MSE | RMSE | MAPE (%) |
|---|---:|---:|---:|---:|
| **LSTM** | **4.5460** | **37.0458** | **6.0865** | **22.9770** |
| GRU | 5.9759 | 57.5909 | 7.5889 | 36.6754 |
| TCN | 10.8209 | 183.8128 | 13.5578 | 72.0659 |

**Best model:** LSTM  
**Best RMSE:** 6.0865 ug/m^3  
**RMSE gap to GRU:** 1.5024 ug/m^3

## Dataset

| Item | Value |
|---|---|
| Source | UK DEFRA Automatic Urban and Rural Network |
| Station | London Marylebone Road |
| Site ID | MY1 |
| File | `MY1_2025.csv` |
| Target | NO2 |
| Input features | CO, PM10, NO, NO2, NOx, O3, PM2.5, SO2 |
| Raw rows | 8,760 |
| Cleaned rows | 7,843 |
| Cleaned range | `2025-01-01 01:00` to `2026-01-01 00:00` |

The original file is a 2025 hourly file. DEFRA records the final hour as `31-12-2025 24:00`, and the code normalizes that timestamp to `2026-01-01 00:00`.

## Method

1. Load the DEFRA CSV after the four metadata rows.
2. Keep eight pollutant columns used by the project.
3. Convert pollutant values to numeric values.
4. Replace negative readings with missing values.
5. Clip each pollutant at the 99th percentile.
6. Fill short gaps with linear interpolation up to six hours.
7. Split the data in chronological order.
8. Scale values with `MinMaxScaler` fitted on training data only.
9. Build 24 hour input sequences.
10. Train LSTM, GRU, and TCN models.
11. Convert predictions back to the original NO2 scale.
12. Save plots and metrics for the Flask dashboard.

## Data Split

| Split | Rows before windowing | Sequences after 24 hour window |
|---|---:|---:|
| Train | 5,490 | 5,466 |
| Validation | 1,176 | 1,152 |
| Test | 1,177 | 1,153 |

The split is chronological, so the test set comes after the training and validation periods.

## Model Summary

| Model | Architecture used in this project |
|---|---|
| LSTM | LSTM 64, LSTM 32, Dense 1 |
| GRU | GRU 64, GRU 32, Dense 1 |
| TCN | Conv1D 64, Conv1D 64 with dilation 2, Conv1D 32 with dilation 4, GlobalAveragePooling1D, Dense 1 |

All models use Adam, MSE loss, early stopping, and a reduce learning rate callback.

## Dashboard Sections

| Section | What it shows |
|---|---|
| Overview | Cleaned rows, input count, lookback window, and best model |
| Data | EDA plots for pollutant readings, March NO2, monthly NO2, and correlation |
| Training | Training and validation loss curves |
| Forecast | Test forecast line plot and predicted versus actual scatter plots |
| Metrics | Saved MAE, MSE, RMSE, and MAPE table |

## Project Files

| File or folder | Purpose |
|---|---|
| `data.py` | Loads, cleans, splits, scales, and windows the data |
| `models.py` | Builds the LSTM, GRU, and TCN models |
| `train.py` | Runs training and saves result plots |
| `app.py` | Flask dashboard app |
| `results.json` | Saved test metrics |
| `static/` | Saved PNG plots used by the dashboard |
| `requirements.txt` | Local training dependencies |
| `requirements-deploy.txt` | Lightweight Flask deployment dependencies |
| `render.yaml` | Render service configuration |

## Run Locally

```bash
git clone https://github.com/abinashprasana/no2-forecasting-deep-learning.git
cd no2-forecasting-deep-learning
pip install -r requirements.txt
python train.py
python app.py
```

Then open:

```text
http://localhost:5000
```

## Deployment

The deployed dashboard runs on Render using the saved images and saved metrics.

```bash
Build Command: pip install -r requirements-deploy.txt
Start Command: gunicorn app:app
```

Only the dashboard dependencies are installed on Render. The deep learning training libraries are kept in `requirements.txt` for local training.

## Limitations

1. The project uses one monitoring station, so the result should not be treated as a general model for all cities.
2. The dataset covers one calendar year, which limits long term seasonal testing.
3. The models use pollutant readings only. Time features such as hour of day and day of week could be tested later.
4. Peak NO2 values are harder to predict, especially after clipping high outliers during preprocessing.
5. The TCN model is intentionally small. A wider dilation setup could be tested in future work.

## Data Source

UK DEFRA Automatic Urban and Rural Network  
https://uk-air.defra.gov.uk

## Author

**Abinash Prasana Selvanathan**

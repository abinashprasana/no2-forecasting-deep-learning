# 🌫️ NO2 Forecasting Dashboard

<p align="center">
  <strong>A Flask dashboard for forecasting hourly NO2 at London Marylebone Road using LSTM, GRU, and TCN models.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-Project-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Training-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-Dashboard-111827?style=for-the-badge&logo=flask&logoColor=white">
  <img alt="Render" src="https://img.shields.io/badge/Render-Free_Service-46E3B7?style=for-the-badge&logo=render&logoColor=111827">
</p>

<p align="center">
  <a href="https://no2-forecasting-dashboard.onrender.com/">
    <img alt="Open Dashboard" src="https://img.shields.io/badge/Open_Dashboard-0EA5E9?style=for-the-badge&logo=render&logoColor=white">
  </a>
</p>

## ✨ What This Project Does

This project predicts the next hourly NO2 value using recent air quality readings from the UK DEFRA AURN London Marylebone Road station.

I cleaned the 2025 hourly dataset, trained three deep learning models, compared their test results, and deployed the saved plots in a Flask dashboard.

The deployed app does not train online. It only serves the saved charts and the saved `results.json` metrics.

## 🏆 Best Result

| Model | MAE | MSE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| 🥇 **LSTM** | **4.5460** | **37.0458** | **6.0865** | **22.9770** |
| GRU | 5.9759 | 57.5909 | 7.5889 | 36.6754 |
| TCN | 10.8209 | 183.8128 | 13.5578 | 72.0659 |

**LSTM performed best with RMSE 6.0865 ug/m^3.**

## 🧠 Architecture Diagram

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"fontFamily": "Inter, Segoe UI, Arial", "primaryTextColor": "#ffffff", "lineColor": "#cbd5e1"}}}%%
flowchart TD
    A["📄 Raw DEFRA CSV<br/>8,760 hourly rows"] --> B["🧹 Data Cleaning<br/>fix 24:00 timestamps<br/>remove negatives<br/>clip 99th percentile<br/>interpolate short gaps"]
    B --> C["📊 Cleaned Dataset<br/>7,843 rows<br/>8 pollutant inputs"]
    C --> D["⏱️ Sequence Window<br/>previous 24 hours<br/>target: next NO2 value"]
    D --> E["🧪 Model Training<br/>LSTM · GRU · TCN"]
    E --> F["📈 Saved Results<br/>plots in static/<br/>metrics in results.json"]
    F --> G["🌐 Flask Dashboard<br/>deployed on Render free service"]

    classDef source fill:#1d4ed8,stroke:#60a5fa,color:#ffffff,stroke-width:2px;
    classDef clean fill:#4f46e5,stroke:#a5b4fc,color:#ffffff,stroke-width:2px;
    classDef model fill:#7c3aed,stroke:#c4b5fd,color:#ffffff,stroke-width:2px;
    classDef deploy fill:#0f766e,stroke:#5eead4,color:#ffffff,stroke-width:2px;

    class A source;
    class B,C,D clean;
    class E,F model;
    class G deploy;
```

## 📸 Dashboard Preview

<p align="center">
  <img src="static/forecast_comparison.png" alt="Forecast comparison plot" width="48%">
  <img src="static/results_comparison.png" alt="Model comparison plot" width="48%">
</p>

## 📌 Dataset

📍 **Station:** London Marylebone Road  
🏷️ **Site ID:** MY1  
📄 **File:** `MY1_2025.csv`  
🧾 **Source:** UK DEFRA Automatic Urban and Rural Network  
🧮 **Raw rows:** 8,760  
✅ **Cleaned rows:** 7,843  
🎯 **Target:** NO2  
🌡️ **Inputs:** CO, PM10, NO, NO2, NOx, O3, PM2.5, SO2

The final `31-12-2025 24:00` record is converted to `2026-01-01 00:00`, which is why the cleaned timestamp range ends at the first hour of 2026.

## 🛠️ What I Used

🐍 Python for the main project code  
🧹 pandas, NumPy, and scikit-learn for cleaning and scaling  
🧠 TensorFlow for LSTM, GRU, and Conv1D TCN models  
📊 Matplotlib for the saved dashboard plots  
🌐 Flask for the dashboard  
🚀 Render free web service for deployment

## 📂 Project Files

📌 `data.py` handles cleaning, scaling, splitting, and 24 hour windows  
📌 `models.py` builds the LSTM, GRU, and TCN models  
📌 `train.py` trains the models and saves the plots  
📌 `app.py` runs the Flask dashboard  
📌 `results.json` stores the final metrics  
📌 `static/` stores the generated chart images  
📌 `render.yaml` keeps the Render deployment setup

## 🚀 Run Locally

```bash
git clone https://github.com/abinashprasana/no2-forecasting-deep-learning.git
cd no2-forecasting-deep-learning
pip install -r requirements.txt
python train.py
python app.py
```

Open:

```text
http://localhost:5000
```

## 🌐 Deployment

The live dashboard is hosted on Render using the saved plots and saved metrics.

```bash
Build Command: pip install -r requirements-deploy.txt
Start Command: gunicorn app:app
```

## 🔍 Notes

1. This is based on one monitoring station, so it should not be treated as a general model for every location.
2. The project uses one year of hourly data.
3. The model predicts the next NO2 value from the previous 24 hours.
4. The LSTM gave the best saved test result in this run.

## 👤 Author

**Abinash Prasana Selvanathan**

Data source: https://uk-air.defra.gov.uk

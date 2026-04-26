import json
from flask import Flask, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NO2 Concentration Forecasting</title>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #34495e;
            --accent-color: #3498db;
            --bg-color: #f4f7f6;
            --text-color: #333;
            --card-bg: #ffffff;
        }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background-color: var(--bg-color); 
            color: var(--text-color); 
            margin: 0; 
            padding: 0; 
            line-height: 1.6;
        }
        .header {
            background-color: var(--primary-color);
            color: white;
            padding: 40px 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 { margin: 0; font-size: 2.5em; font-weight: 600; }
        .header p { margin-top: 10px; font-size: 1.2em; opacity: 0.9; }
        .container { 
            max-width: 1100px; 
            margin: 40px auto; 
            padding: 0 20px; 
        }
        .section { 
            background: var(--card-bg); 
            padding: 30px; 
            margin-bottom: 40px; 
            border-radius: 10px; 
            box-shadow: 0 5px 15px rgba(0,0,0,0.05); 
            transition: transform 0.2s ease-in-out;
        }
        .section:hover { transform: translateY(-5px); }
        h2 { 
            color: var(--primary-color); 
            border-bottom: 3px solid var(--accent-color); 
            padding-bottom: 10px; 
            margin-top: 0; 
            font-size: 1.8em;
        }
        .description {
            font-size: 1.1em;
            color: #555;
            margin-bottom: 25px;
            background-color: #eaf2f8;
            padding: 15px;
            border-left: 5px solid var(--accent-color);
            border-radius: 4px;
        }
        .img-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }
        .img-grid.two-cols { grid-template-columns: 1fr 1fr; }
        .img-container { 
            text-align: center; 
            background: #fff;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #eee;
        }
        .img-container img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 4px; 
        }
        .img-caption {
            margin-top: 10px;
            font-size: 0.9em;
            color: #777;
            font-style: italic;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        th, td { 
            padding: 15px; 
            border: 1px solid #eee; 
            text-align: center; 
            font-size: 1.1em; 
        }
        th { 
            background-color: var(--secondary-color); 
            color: white; 
            font-weight: 600; 
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        .highlight { font-weight: bold; color: var(--accent-color); }
        .footer {
            text-align: center;
            padding: 20px;
            color: #777;
            margin-top: 40px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>NO2 Concentration Forecasting Using Deep Learning: London Marylebone Road (2025)</h1>
    </div>

    <div class="container">
        
        <div class="section">
            <h2>1. Exploratory Data Analysis</h2>
            <div class="description">
                <p>Before building the deep learning models, I analyzed the UK DEFRA AURN dataset for 2025. The data required cleaning to remove instrument artefacts like negative values and extreme outliers. We used linear interpolation to fill in short gaps.</p>
                <p>The correlation heatmap is useful to understand multicollinearity before modelling. I specifically looked at March 2025 for the NO2 closeup because it sits in the middle of the year, avoiding summer low pollution months and winter peak months, giving a representative view of a typical daily cycle.</p>
            </div>
            
            <div class="img-container">
                <img src="{{ url_for('static', filename='eda_all_pollutants.png') }}" alt="All Pollutants">
                <div class="img-caption">Overview of all 8 pollutants across the year</div>
            </div>
            
            <div class="img-grid two-cols" style="margin-top: 20px;">
                <div class="img-container">
                    <img src="{{ url_for('static', filename='eda_no2_march.png') }}" alt="NO2 March 2025">
                    <div class="img-caption">Typical daily NO2 cycle in March</div>
                </div>
                <div class="img-container">
                    <img src="{{ url_for('static', filename='eda_monthly_avg.png') }}" alt="Monthly Average">
                    <div class="img-caption">Monthly average NO2 showing seasonal patterns</div>
                </div>
            </div>
            
            <div class="img-container" style="margin-top: 20px;">
                <img src="{{ url_for('static', filename='eda_correlation.png') }}" alt="Correlation Matrix">
                <div class="img-caption">Feature Correlation Matrix</div>
            </div>
        </div>

        <div class="section">
            <h2>2. Model Training and Validation</h2>
            <div class="description">
                <p>I prepared the data using a chronological 70 / 15 / 15 split for training, validation, and testing to prevent data leakage. I used a 24 hour sliding window approach so each sample uses 24 past hours to predict the next hour's NO2 concentration.</p>
                <p>I built three sequence models for comparison: an LSTM, a GRU, and a Temporal Convolutional Network (TCN). The GRU model has fewer parameters than the LSTM so the overfitting risk is lower. During training, it is normal in Keras for the validation loss to sit slightly below the training loss because training computes loss per mini batch while validation evaluates the full set.</p>
            </div>
            
            <div class="img-container">
                <img src="{{ url_for('static', filename='training_loss_curves.png') }}" alt="Training Loss Curves">
                <div class="img-caption">Training versus Validation Loss across Epochs</div>
            </div>
        </div>

        <div class="section">
            <h2>3. Forecasting Results</h2>
            <div class="description">
                <p>After training, I evaluated the models on the unseen test set. The forecast overlay below shows the first 200 test hours to keep the curves readable. The scatter plots show the predicted versus actual values, with the red line indicating a perfect prediction.</p>
            </div>
            
            <div class="img-container">
                <img src="{{ url_for('static', filename='forecast_comparison.png') }}" alt="Forecast Comparison">
                <div class="img-caption">NO2 Forecast versus Actual for the first 200 test hours</div>
            </div>
            
            <div class="img-container" style="margin-top: 20px;">
                <img src="{{ url_for('static', filename='forecast_scatter.png') }}" alt="Predicted vs Actual">
                <div class="img-caption">Scatter plots of predictions on the test set</div>
            </div>
        </div>

        <div class="section">
            <h2>4. Final Evaluation Metrics</h2>
            <div class="description">
                <p>The final performance of each model on the test set was measured using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). The predictions were inverse transformed to the original scale (µg/m³) to make the errors interpretable.</p>
            </div>
            
            <table>
                <tr>
                    <th>Model</th>
                    <th>MAE</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th>MAPE (%)</th>
                </tr>
                {% for model, metrics in results.items() %}
                <tr>
                    <td class="highlight">{{ model }}</td>
                    <td>{{ metrics.MAE }}</td>
                    <td>{{ metrics.MSE }}</td>
                    <td>{{ metrics.RMSE }}</td>
                    <td>{{ metrics.MAPE }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <div class="img-container" style="margin-top: 20px;">
                <img src="{{ url_for('static', filename='results_comparison.png') }}" alt="Model Comparison">
                <div class="img-caption">Bar chart comparison of model metrics</div>
            </div>
        </div>

        <div class="footer">
            <p>Project Dashboard &bull; NO2 Forecasting Analysis</p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {"Error": {"MAE": "N/A", "MSE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}}
        
    return render_template_string(HTML_TEMPLATE, results=results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

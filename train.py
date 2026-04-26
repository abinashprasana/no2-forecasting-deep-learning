import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data import load_and_clean_data, prepare_data
from models import build_lstm, build_gru, build_tcn, compute_metrics

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def save_eda_plots(df, static_dir):
    # All Pollutants
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()
    colours = ['steelblue','darkorange','seagreen','crimson',
               'mediumpurple','saddlebrown','teal','goldenrod']
    for i, col in enumerate(df.columns):
        axes[i].plot(df.index, df[col], linewidth=0.4, color=colours[i])
        axes[i].set_title(col, fontsize=10, fontweight='bold')
        axes[i].set_ylabel('µg/m³' if col != 'CO' else 'mg/m³')
        axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.suptitle('London Marylebone Road — All Pollutants (2025)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'eda_all_pollutants.png'))
    plt.close()

    # NO2 closeup
    sample = df['NO2'].loc['2025-03-01':'2025-03-31']
    plt.figure(figsize=(13, 4))
    plt.plot(sample.index, sample.values, color='steelblue', linewidth=0.9)
    plt.title('NO2 — March 2025 (hourly)', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('NO2 (µg/m³)')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'eda_no2_march.png'))
    plt.close()

    # Monthly average NO2
    monthly_avg = df['NO2'].resample('ME').mean()
    plt.figure(figsize=(10, 4))
    plt.bar(monthly_avg.index.strftime('%b'), monthly_avg.values,
            color='steelblue', edgecolor='black', linewidth=0.6)
    plt.title('Average NO2 by Month — 2025', fontsize=12)
    plt.xlabel('Month')
    plt.ylabel('Mean NO2 (µg/m³)')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'eda_monthly_avg.png'))
    plt.close()

    # Correlation matrix
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(df.columns, fontsize=9)
    ax.set_title('Feature Correlation Matrix', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'eda_correlation.png'))
    plt.close()

def save_training_plots(histories, static_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h.history['loss'], label='Train', color='steelblue')
        ax.plot(h.history['val_loss'], label='Val', color='darkorange', linestyle='--')
        ax.set_title(f'{name} — Loss Curves', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
    plt.suptitle('Training vs Validation Loss — All Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'training_loss_curves.png'))
    plt.close()

def save_forecast_plots(y_true_real, preds_dict, static_dir):
    n_show = 200
    plt.figure(figsize=(14, 5))
    plt.plot(y_true_real[:n_show], label='Actual NO2', color='black', linewidth=1.2)
    plt.plot(preds_dict['LSTM'][:n_show], label='LSTM', color='steelblue', linewidth=1.0, linestyle='--')
    plt.plot(preds_dict['GRU'][:n_show], label='GRU', color='darkorange', linewidth=1.0, linestyle='-.')
    plt.plot(preds_dict['TCN'][:n_show], label='TCN', color='seagreen', linewidth=1.0, linestyle=':')
    plt.title('NO2 Forecast vs Actual — First 200 Test Hours', fontsize=12)
    plt.xlabel('Time Step (hours)')
    plt.ylabel('NO2 Concentration (µg/m³)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'forecast_comparison.png'))
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colours = {'LSTM': 'steelblue', 'GRU': 'darkorange', 'TCN': 'seagreen'}
    for ax, (name, preds) in zip(axes, preds_dict.items()):
        ax.scatter(y_true_real, preds, alpha=0.2, s=5, color=colours[name])
        lo = min(y_true_real.min(), preds.min())
        hi = max(y_true_real.max(), preds.max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='Perfect fit')
        ax.set_title(f'{name} — Predicted vs Actual', fontsize=11)
        ax.set_xlabel('Actual NO2 (µg/m³)')
        ax.set_ylabel('Predicted NO2 (µg/m³)')
        ax.legend(fontsize=8)
    plt.suptitle('Predicted vs Actual NO2 — Test Set', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'forecast_scatter.png'))
    plt.close()

def save_results_plot(results, static_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    colours = ['steelblue', 'darkorange', 'seagreen']
    models  = list(results.keys())
    
    for ax, metric in zip(axes, ['MAE', 'MSE', 'RMSE', 'MAPE']):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(models, vals, color=colours, edgecolor='black', linewidth=0.6, width=0.45)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(vals) * 1.18)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
    plt.suptitle('Model Comparison — Test Set (NO2 Forecasting)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'results_comparison.png'))
    plt.close()

def main():
    static_dir = 'static'
    os.makedirs(static_dir, exist_ok=True)
    
    print("Loading and cleaning data...")
    df = load_and_clean_data('MY1_2025.csv')
    
    print("Saving EDA plots...")
    save_eda_plots(df, static_dir)
    
    seq_len = 24
    print("Preparing sequences and train/val/test splits...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, target_idx = prepare_data(df, target_col='NO2', seq_len=seq_len)
    
    n_features = X_train.shape[2]
    
    models = {
        'LSTM': build_lstm(seq_len, n_features),
        'GRU': build_gru(seq_len, n_features),
        'TCN': build_tcn(seq_len, n_features)
    }
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
    ]
    
    histories = {}
    preds_dict = {}
    results = {}
    y_true_real = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        histories[name] = hist
        
        preds_sc = model.predict(X_test, verbose=0).flatten()
        metrics, y_true, preds_real = compute_metrics(y_test, preds_sc, scaler, target_idx)
        
        preds_dict[name] = preds_real
        results[name] = metrics
        
        if y_true_real is None:
            y_true_real = y_true
            
        print(f"{name} Test Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
            
    print("\nSaving training curves...")
    save_training_plots(histories, static_dir)
    
    print("Saving forecast comparison plots...")
    save_forecast_plots(y_true_real, preds_dict, static_dir)
    
    print("Saving results plot...")
    save_results_plot(results, static_dir)
    
    print("Saving results to results.json...")
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    best_model = min(results, key=lambda m: results[m]['RMSE'])
    print(f"\nBest model by RMSE: {best_model}")
    print("\nTraining complete. Results and plots saved to the static/ directory.")

if __name__ == '__main__':
    main()

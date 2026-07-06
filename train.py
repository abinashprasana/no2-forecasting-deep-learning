import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data import load_and_clean_data, prepare_data
from models import build_gru, build_lstm, build_tcn, compute_metrics

tf.random.set_seed(42)
np.random.seed(42)

STATIC_DIR = Path("static")
SEQ_LEN = 24

MODEL_COLORS = {
    "LSTM": "#1f77b4",
    "GRU": "#ff8c00",
    "TCN": "#2e8b57",
}

POLLUTANT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def save_figure(fig, path, rect=None):
    if rect:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()

    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_eda_plots(df):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]
        ax.plot(df.index, df[col], linewidth=0.6, color=POLLUTANT_COLORS[i])
        ax.set_title(col)
        ax.set_ylabel("ug/m^3" if col != "CO" else "mg/m^3")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, alpha=0.25)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    for ax in axes[len(df.columns) :]:
        ax.axis("off")

    fig.suptitle("London Marylebone Road pollutants, 2025")
    save_figure(fig, STATIC_DIR / "eda_all_pollutants.png", rect=[0, 0, 1, 0.97])

    sample = df["NO2"].loc["2025-03-01":"2025-03-31"]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(sample.index, sample.values, color=MODEL_COLORS["LSTM"], linewidth=1)
    ax.set_title("NO2, March 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("NO2 (ug/m^3)")
    ax.grid(True, alpha=0.25)
    save_figure(fig, STATIC_DIR / "eda_no2_march.png")

    monthly_avg = df["NO2"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly_avg.index.strftime("%b"), monthly_avg.values, color=MODEL_COLORS["GRU"])
    ax.set_title("Average NO2 by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean NO2 (ug/m^3)")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, STATIC_DIR / "eda_monthly_avg.png")

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.colorbar(ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1), ax=ax)
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.columns)
    ax.set_title("Feature correlation")
    save_figure(fig, STATIC_DIR / "eda_correlation.png")


def save_training_plots(histories):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history.history["loss"], label="Train", color=MODEL_COLORS[name])
        ax.plot(history.history["val_loss"], label="Validation", color="#d62728", linestyle="--")
        ax.set_title(f"{name} loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Training and validation loss")
    save_figure(fig, STATIC_DIR / "training_loss_curves.png", rect=[0, 0, 1, 0.94])


def save_forecast_plots(y_true, predictions):
    n_show = 200

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true[:n_show], label="Actual NO2", color="black", linewidth=1.2)
    ax.plot(predictions["LSTM"][:n_show], label="LSTM", color=MODEL_COLORS["LSTM"], linestyle="--")
    ax.plot(predictions["GRU"][:n_show], label="GRU", color=MODEL_COLORS["GRU"], linestyle="-.")
    ax.plot(predictions["TCN"][:n_show], label="TCN", color=MODEL_COLORS["TCN"], linestyle=":")
    ax.set_title("NO2 forecast vs actual, first 200 test hours")
    ax.set_xlabel("Test hour")
    ax.set_ylabel("NO2 (ug/m^3)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, STATIC_DIR / "forecast_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (name, preds) in zip(axes, predictions.items()):
        low = min(y_true.min(), preds.min())
        high = max(y_true.max(), preds.max())

        ax.scatter(y_true, preds, alpha=0.3, s=7, color=MODEL_COLORS[name])
        ax.plot([low, high], [low, high], color="#d62728", linestyle="--", label="Perfect fit")
        ax.set_title(f"{name}: predicted vs actual")
        ax.set_xlabel("Actual NO2 (ug/m^3)")
        ax.set_ylabel("Predicted NO2 (ug/m^3)")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Predicted and actual NO2 on the test set")
    save_figure(fig, STATIC_DIR / "forecast_scatter.png", rect=[0, 0, 1, 0.94])


def save_results_plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    models = list(results)

    for ax, metric in zip(axes, ["MAE", "MSE", "RMSE", "MAPE"]):
        values = [results[model][metric] for model in models]
        colors = [MODEL_COLORS[model] for model in models]
        bars = ax.bar(models, values, color=colors)

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(True, axis="y", alpha=0.25)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Model comparison on the test set")
    save_figure(fig, STATIC_DIR / "results_comparison.png", rect=[0, 0, 1, 0.96])


def callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
    ]


def main():
    STATIC_DIR.mkdir(exist_ok=True)

    print("load data")
    df = load_and_clean_data("MY1_2025.csv")

    print("save data plots")
    save_eda_plots(df)

    print("prepare data")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, target_idx = prepare_data(
        df,
        target_col="NO2",
        seq_len=SEQ_LEN,
    )

    n_features = X_train.shape[2]
    models = {
        "LSTM": build_lstm(SEQ_LEN, n_features),
        "GRU": build_gru(SEQ_LEN, n_features),
        "TCN": build_tcn(SEQ_LEN, n_features),
    }

    histories = {}
    predictions = {}
    results = {}
    y_true_real = None

    for name, model in models.items():
        print(f"train {name}")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=callbacks(),
            verbose=0,
        )
        histories[name] = history

        preds_scaled = model.predict(X_test, verbose=0).flatten()
        metrics, y_true, preds_real = compute_metrics(y_test, preds_scaled, scaler, target_idx)

        predictions[name] = preds_real
        results[name] = metrics
        if y_true_real is None:
            y_true_real = y_true

        print(
            f"{name}: RMSE={metrics['RMSE']}, "
            f"MAE={metrics['MAE']}, MAPE={metrics['MAPE']}"
        )

    print("save model plots")
    save_training_plots(histories)
    save_forecast_plots(y_true_real, predictions)
    save_results_plot(results)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    best_model = min(results, key=lambda model: results[model]["RMSE"])
    print(f"best model: {best_model}")


if __name__ == "__main__":
    main()

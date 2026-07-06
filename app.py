import json
import math
from pathlib import Path

from flask import Flask, render_template_string

app = Flask(__name__)

RESULTS_PATH = Path("results.json")

PLOT_SECTIONS = [
    {
        "id": "data",
        "number": "1",
        "title": "Data checks",
        "summary": (
            "The cleaned file keeps eight pollutant inputs from London Marylebone Road. "
            "Negative readings are treated as missing, short gaps are interpolated, "
            "and high outliers are clipped before scaling."
        ),
        "plots": [
            {
                "file": "eda_all_pollutants.png",
                "title": "All pollutants across 2025",
                "note": "Full year view of the eight inputs.",
                "size": "full",
            },
            {
                "file": "eda_no2_march.png",
                "title": "NO2 in March",
                "note": "Hourly NO2 close-up.",
                "size": "half",
            },
            {
                "file": "eda_monthly_avg.png",
                "title": "Monthly NO2 average",
                "note": "Seasonal movement in the target.",
                "size": "half",
            },
            {
                "file": "eda_correlation.png",
                "title": "Feature correlation",
                "note": "Relationship between pollutant inputs.",
                "size": "full",
            },
        ],
    },
    {
        "id": "training",
        "number": "2",
        "title": "Training",
        "summary": (
            "The split is chronological: 70 percent training, 15 percent validation, "
            "and 15 percent test. Each sample uses the previous 24 hours to forecast "
            "the next NO2 value."
        ),
        "plots": [
            {
                "file": "training_loss_curves.png",
                "title": "Training and validation loss",
                "note": "LSTM, GRU, and TCN loss curves.",
                "size": "full",
            }
        ],
    },
    {
        "id": "forecast",
        "number": "3",
        "title": "Forecast",
        "summary": (
            "The forecast plots use the held-out test set. The line chart shows the "
            "first 200 test hours, and the scatter plot compares predicted and actual "
            "NO2 values."
        ),
        "plots": [
            {
                "file": "forecast_comparison.png",
                "title": "Forecast vs actual",
                "note": "First 200 test hours.",
                "size": "full",
            },
            {
                "file": "forecast_scatter.png",
                "title": "Predicted vs actual",
                "note": "Test set scatter plots.",
                "size": "full",
            },
        ],
    },
]

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NO2 Forecasting</title>
  <style>
    :root {
      --bg: #08111f;
      --bg-soft: #0d2234;
      --surface: #10263a;
      --surface-2: #153149;
      --line: #2e5369;
      --text: #f5f9fc;
      --muted: #bfd0dd;
      --teal: #2dd4bf;
      --sky: #38bdf8;
      --amber: #fbbf24;
      --rose: #fb7185;
      --paper: #f8fafc;
      --paper-edge: #dce8f2;
      --ink: #111827;
    }

    * {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      margin: 0;
      color: var(--text);
      background:
        linear-gradient(120deg, rgba(45, 212, 191, .10), transparent 30%),
        linear-gradient(280deg, rgba(251, 191, 36, .09), transparent 32%),
        linear-gradient(180deg, var(--bg), var(--bg-soft) 48%, var(--bg));
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }

    a {
      color: inherit;
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid rgba(46, 83, 105, .8);
      background: rgba(6, 15, 28, .92);
      backdrop-filter: blur(14px);
    }

    .topbar-inner,
    main {
      width: min(1380px, calc(100% - 44px));
      margin: 0 auto;
    }

    .topbar-inner {
      min-height: 70px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 22px;
    }

    .brand {
      font-size: 1.02rem;
      font-weight: 850;
      letter-spacing: .01em;
    }

    .brand span {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: .82rem;
      font-weight: 550;
    }

    nav {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      white-space: nowrap;
    }

    nav a {
      color: var(--muted);
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 8px 11px;
      text-decoration: none;
      font-size: .9rem;
      font-weight: 750;
    }

    nav a:hover {
      color: var(--text);
      border-color: rgba(56, 189, 248, .35);
      background: rgba(56, 189, 248, .10);
    }

    main {
      padding: 30px 0 60px;
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(360px, .75fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 42px;
    }

    .hero-copy,
    .stat,
    .plot,
    .table-wrap,
    .result-note {
      border: 1px solid rgba(94, 139, 166, .48);
      border-radius: 8px;
      background: rgba(16, 38, 58, .86);
      box-shadow: 0 20px 45px rgba(0, 0, 0, .22);
    }

    .hero-copy {
      min-height: 340px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 38px;
      background:
        linear-gradient(135deg, rgba(56, 189, 248, .16), transparent 36%),
        linear-gradient(150deg, rgba(16, 38, 58, .95), rgba(21, 49, 73, .92));
    }

    .kicker,
    .label,
    th {
      color: var(--muted);
      text-transform: uppercase;
      font-size: .76rem;
      font-weight: 850;
      letter-spacing: .06em;
    }

    h1,
    h2,
    h3,
    p,
    figure {
      margin-top: 0;
    }

    h1 {
      max-width: 850px;
      margin-bottom: 16px;
      font-size: 4rem;
      line-height: .98;
    }

    h2 {
      margin-bottom: 10px;
      font-size: 2rem;
      line-height: 1.12;
    }

    .lead {
      max-width: 760px;
      margin-bottom: 0;
      color: var(--muted);
      font-size: 1.08rem;
    }

    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .stat {
      min-height: 160px;
      padding: 18px;
      background: linear-gradient(180deg, rgba(21, 49, 73, .94), rgba(13, 34, 52, .94));
    }

    .stat:nth-child(2) {
      border-color: rgba(45, 212, 191, .5);
    }

    .stat:nth-child(4) {
      border-color: rgba(251, 191, 36, .52);
    }

    .value {
      margin: 12px 0 8px;
      font-size: 2.15rem;
      line-height: 1;
      font-weight: 900;
    }

    .hint {
      margin: 0;
      color: var(--muted);
      font-size: .94rem;
    }

    .section {
      scroll-margin-top: 96px;
      margin-bottom: 50px;
    }

    .section-head {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 16px;
      align-items: start;
      margin-bottom: 18px;
    }

    .section-number {
      display: grid;
      width: 42px;
      height: 42px;
      place-items: center;
      border-radius: 8px;
      color: var(--ink);
      background: linear-gradient(135deg, var(--teal), var(--amber));
      font-weight: 900;
    }

    .section-copy {
      max-width: 920px;
      margin-bottom: 0;
      color: var(--muted);
    }

    .plot-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .plot {
      margin: 0;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(21, 49, 73, .94), rgba(10, 29, 46, .96));
    }

    .plot.full {
      grid-column: 1 / -1;
    }

    .plot-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 18px;
      padding: 15px 18px;
      border-bottom: 1px solid rgba(94, 139, 166, .45);
    }

    .plot-head strong {
      font-size: 1.02rem;
    }

    .plot-head span {
      color: var(--muted);
      font-size: .9rem;
      text-align: right;
    }

    .plot-body {
      padding: 14px;
      background:
        linear-gradient(180deg, rgba(248, 250, 252, .96), rgba(236, 244, 250, .96));
    }

    .plot img {
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--paper-edge);
      border-radius: 6px;
      background: var(--paper);
    }

    .metrics-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      align-items: start;
      margin-top: 18px;
    }

    .table-wrap {
      overflow-x: auto;
      background: rgba(16, 38, 58, .94);
    }

    table {
      width: 100%;
      min-width: 680px;
      border-collapse: collapse;
    }

    th,
    td {
      padding: 16px 18px;
      border-bottom: 1px solid rgba(94, 139, 166, .35);
      text-align: right;
      white-space: nowrap;
    }

    th:first-child,
    td:first-child {
      text-align: left;
    }

    tr:last-child td {
      border-bottom: 0;
    }

    .model {
      font-weight: 900;
    }

    .badge {
      display: inline-block;
      margin-left: 8px;
      padding: 3px 8px;
      border-radius: 999px;
      color: var(--ink);
      background: var(--amber);
      font-size: .72rem;
      font-weight: 900;
    }

    .bar {
      display: inline-block;
      width: 96px;
      height: 6px;
      margin-left: 10px;
      border-radius: 999px;
      background: rgba(191, 208, 221, .18);
      vertical-align: middle;
    }

    .bar span {
      display: block;
      width: var(--w);
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--teal), var(--sky));
    }

    .result-note {
      padding: 20px;
      background:
        linear-gradient(135deg, rgba(45, 212, 191, .18), transparent 50%),
        linear-gradient(180deg, rgba(21, 49, 73, .96), rgba(16, 38, 58, .96));
    }

    .result-note strong {
      display: block;
      margin-bottom: 10px;
      font-size: 1.22rem;
      line-height: 1.2;
    }

    footer {
      margin-top: 54px;
      padding-top: 18px;
      border-top: 1px solid rgba(94, 139, 166, .38);
      color: var(--muted);
      font-size: .94rem;
    }

    @media (max-width: 1040px) {
      .hero,
      .metrics-layout {
        grid-template-columns: 1fr;
      }

      .stats {
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
    }

    @media (max-width: 820px) {
      .topbar-inner {
        align-items: flex-start;
        flex-direction: column;
        padding: 13px 0;
      }

      nav {
        width: 100%;
      }

      .hero-copy {
        min-height: 0;
        padding: 28px;
      }

      .stats,
      .plot-grid {
        grid-template-columns: 1fr;
      }

      .plot.full {
        grid-column: auto;
      }

      .plot-head {
        align-items: flex-start;
        flex-direction: column;
        gap: 5px;
      }

      .plot-head span {
        text-align: left;
      }
    }

    @media (max-width: 560px) {
      .topbar-inner,
      main {
        width: min(1380px, calc(100% - 28px));
      }

      nav {
        flex-wrap: wrap;
        gap: 6px;
        overflow: visible;
        white-space: normal;
      }

      nav a {
        flex: 1 1 auto;
        padding: 7px 8px;
        text-align: center;
      }

      h1 {
        font-size: 2.55rem;
      }

      h2 {
        font-size: 1.55rem;
      }

      .value {
        font-size: 1.8rem;
      }

      .stat {
        min-height: 126px;
      }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">NO2 Forecasting <span>London Marylebone Road</span></div>
      <nav aria-label="Sections">
        <a href="#overview">Overview</a>
        <a href="#data">Data</a>
        <a href="#training">Training</a>
        <a href="#forecast">Forecast</a>
        <a href="#metrics">Metrics</a>
      </nav>
    </div>
  </header>

  <main>
    <section id="overview" class="hero">
      <div class="hero-copy">
        <p class="kicker">2025 hourly data</p>
        <h1>NO2 Forecast Dashboard</h1>
        <p class="lead">Cleaned DEFRA AURN data, a 24-hour lookback window, and three sequence models: LSTM, GRU, and TCN.</p>
      </div>

      <div class="stats">
        {% for card in cards %}
        <article class="stat">
          <p class="label">{{ card.label }}</p>
          <p class="value">{{ card.value }}</p>
          <p class="hint">{{ card.note }}</p>
        </article>
        {% endfor %}
      </div>
    </section>

    {% for section in plot_sections %}
    <section id="{{ section.id }}" class="section">
      <div class="section-head">
        <div class="section-number">{{ section.number }}</div>
        <div>
          <h2>{{ section.title }}</h2>
          <p class="section-copy">{{ section.summary }}</p>
        </div>
      </div>

      <div class="plot-grid">
        {% for plot in section.plots %}
        <figure class="plot {{ plot.size }}">
          <figcaption class="plot-head">
            <strong>{{ plot.title }}</strong>
            <span>{{ plot.note }}</span>
          </figcaption>
          <div class="plot-body">
            <img src="{{ url_for('static', filename=plot.file) }}" alt="{{ plot.title }}">
          </div>
        </figure>
        {% endfor %}
      </div>
    </section>
    {% endfor %}

    <section id="metrics" class="section">
      <div class="section-head">
        <div class="section-number">4</div>
        <div>
          <h2>Final metrics</h2>
          <p class="section-copy">The saved test metrics are shown on the original scale. Lower values are better for MAE, MSE, RMSE, and MAPE.</p>
        </div>
      </div>

      <figure class="plot full">
        <figcaption class="plot-head">
          <strong>Model comparison</strong>
          <span>Saved test metrics.</span>
        </figcaption>
        <div class="plot-body">
          <img src="{{ url_for('static', filename='results_comparison.png') }}" alt="Model comparison">
        </div>
      </figure>

      <div class="metrics-layout">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>MSE</th>
                <th>RMSE</th>
                <th>MAPE (%)</th>
              </tr>
            </thead>
            <tbody>
              {% for row in ranked %}
              <tr>
                <td class="model">
                  {{ row.model }}
                  {% if row.model == best_model %}<span class="badge">Best</span>{% endif %}
                </td>
                <td>{{ row.MAE }}</td>
                <td>{{ row.MSE }}</td>
                <td>
                  {{ row.RMSE }}
                  <span class="bar" aria-hidden="true"><span style="--w: {{ row.rmse_width }}%;"></span></span>
                </td>
                <td>{{ row.MAPE }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>

        <aside class="result-note">
          <strong>{{ best_model }} has the lowest RMSE</strong>
          <p class="hint">{{ result_note }}</p>
        </aside>
      </div>
    </section>

    <footer>DEFRA AURN London Marylebone Road. Target: NO2.</footer>
  </main>
</body>
</html>
"""


def load_results():
    try:
        return json.loads(RESULTS_PATH.read_text())
    except FileNotFoundError:
        return {}


def is_complete(metrics):
    return all(
        isinstance(metrics.get(name), (int, float)) and math.isfinite(metrics[name])
        for name in ("MAE", "MSE", "RMSE", "MAPE")
    )


def view_model(results):
    rows = [
        {"model": model, **metrics}
        for model, metrics in results.items()
        if is_complete(metrics)
    ]
    rows.sort(key=lambda row: row["RMSE"])

    if not rows:
        empty = {"model": "N/A", "MAE": "N/A", "MSE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}
        return [empty], "N/A", "No saved numeric results found."

    max_rmse = max(row["RMSE"] for row in rows)
    for row in rows:
        row["rmse_width"] = f"{(row['RMSE'] / max_rmse) * 100:.2f}"

    best = rows[0]
    if len(rows) == 1:
        note = "Only one model result is saved."
    else:
        gap = rows[1]["RMSE"] - best["RMSE"]
        note = f"RMSE {best['RMSE']} is {gap:.4f} lower than {rows[1]['model']}."

    return rows, best["model"], note


@app.route("/")
def index():
    ranked, best_model, result_note = view_model(load_results())
    best = ranked[0]
    cards = [
        {"label": "Cleaned rows", "value": "7,843", "note": "hourly records kept"},
        {"label": "Inputs", "value": "8", "note": "pollutant features"},
        {"label": "Window", "value": "24h", "note": "lookback sequence"},
        {"label": "Best model", "value": best_model, "note": f"RMSE {best['RMSE']}"},
    ]
    return render_template_string(
        HTML,
        cards=cards,
        plot_sections=PLOT_SECTIONS,
        ranked=ranked,
        best_model=best_model,
        result_note=result_note,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

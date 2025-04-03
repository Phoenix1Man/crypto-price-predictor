# dashboard/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo

from utils.data_fetcher import get_historical_price, get_available_coins
from utils.preprocessing import (
    scale_data,
    attach_sentiment,
    create_multivariate_sequences,
    train_test_split
)
from models.lstm_model import build_lstm_model, train_lstm_model, predict_and_inverse
from sentiment.analyzer import get_news_headlines, analyze_sentiment, get_daily_sentiment_map

import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    available_coins = get_available_coins()
    selected_days = int(request.form.get("days") or 60)
    selected_coins = request.form.getlist("coins") or ["bitcoin", "ethereum", "binancecoin"]

    charts = []
    sentiment_summary = {}

    for coin in selected_coins:
        try:
            print(f"\n=== Processing {coin.upper()} ===")

            df = get_historical_price(coin, days=selected_days)
            sentiment_map = get_daily_sentiment_map()
            df = attach_sentiment(df, sentiment_map)

            scaled_price, scaler = scale_data(df)
            sentiment_values = df["sentiment"].values.reshape(-1, 1)
            X, y = create_multivariate_sequences(scaled_price, sentiment_values, window_size=60)

            if len(X) < 5:
                print(f"[{coin}] Not enough data to train (X len = {len(X)})")
                charts.append({"coin": coin, "chart": f"<p>Not enough data to train model for {coin}. Try increasing 'days'.</p>"})
                sentiment_summary[coin] = "N/A"
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            print(f"[{coin}] X_test shape: {X_test.shape}")
            print(f"[{coin}] y_test shape: {y_test.shape}")

            model = build_lstm_model((X_train.shape[1], 2))
            model, _ = train_lstm_model(model, X_train, y_train, epochs=5)

            predicted_prices = predict_and_inverse(model, X_test, scaler)
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            print(f"[{coin}] Predicted shape: {predicted_prices.shape}")
            print(f"[{coin}] Predicted sample: {predicted_prices[:5].flatten()}")
            print(f"[{coin}] Actual sample: {y_test_rescaled[:5].flatten()}")
            print(f"[{coin}] NaN in prediction? {np.isnan(predicted_prices).any()}")
            print(f"[{coin}] NaN in actual? {np.isnan(y_test_rescaled).any()}")

            if len(predicted_prices) < 2 or np.isnan(predicted_prices).any():
                print(f"[{coin}] Skipping plot due to invalid prediction data.")
                charts.append({"coin": coin, "chart": f"<p>No valid prediction data for {coin}.</p>"})
                sentiment_summary[coin] = "N/A"
                continue

            actual_y = [float(v) for v in y_test_rescaled.flatten()]
            predicted_y = [float(v) for v in predicted_prices.flatten()]

            headlines = get_news_headlines()
            sentiment_scores = analyze_sentiment([h[0] for h in headlines])
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            sentiment_label = (
                "Positive" if avg_sentiment > 0.2
                else "Negative" if avg_sentiment < -0.2
                else "Neutral"
            )
            sentiment_summary[coin] = sentiment_label

            trace_actual = go.Scatter(y=actual_y, mode='lines', name='Actual Price')
            trace_predicted = go.Scatter(y=predicted_y, mode='lines', name='Predicted Price')

            layout = go.Layout(
                title=f"{coin.capitalize()} Price Prediction",
                xaxis_title="Time Step",
                yaxis_title="Price (USD)",
                plot_bgcolor="#1e1e1e",
                paper_bgcolor="#1e1e1e",
                font=dict(color="white"),
                legend=dict(orientation="h", x=0.3, y=-0.2)
            )

            fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
            chart_html = pyo.plot(fig, output_type="div", include_plotlyjs=False)

            charts.append({"coin": coin, "chart": chart_html})

        except Exception as e:
            print(f"[ERROR] Failed for {coin}: {e}")
            charts.append({"coin": coin, "chart": f"<p>Error loading {coin}: {e}</p>"})
            sentiment_summary[coin] = "N/A"

    # ✅ Fetch news once and pass it
    headlines_raw = get_news_headlines()
    news_headlines = [h[0] for h in headlines_raw]

    return render_template(
        "index.html",
        charts=charts,
        coins=selected_coins,
        available_coins=available_coins,
        selected_days=selected_days,
        sentiment_summary=sentiment_summary,
        news_headlines=news_headlines
    )

if __name__ == "__main__":
    app.run(debug=True)

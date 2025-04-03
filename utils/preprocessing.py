# utils/preprocessing.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def plot_prices(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["price"], label="Price")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.title("Crypto Price History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df["price"].values.reshape(-1, 1))
    return scaled, scaler

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_test_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def attach_sentiment(df, sentiment_map):
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["sentiment"] = df["date"].map(sentiment_map).fillna(0.0)
    return df

def create_multivariate_sequences(prices, sentiments, window_size=60):
    X, y = [], []
    for i in range(window_size, len(prices)):
        price_seq = prices[i - window_size:i]
        sentiment_seq = sentiments[i - window_size:i].reshape(-1, 1)
        combined = np.concatenate((price_seq, sentiment_seq), axis=1)
        X.append(combined)
        y.append(prices[i])
    return np.array(X), np.array(y)
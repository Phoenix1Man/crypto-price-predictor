# main.py

from utils.data_fetcher import get_historical_price
from utils.preprocessing import plot_prices, scale_data, create_sequences, train_test_split
from models.lstm_model import build_lstm_model, train_lstm_model, predict_and_inverse

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Load historical price data
    df = get_historical_price("bitcoin", days=30)

    # 2. Visualize raw prices
    plot_prices(df)

    # 3. Scale prices
    scaled_data, scaler = scale_data(df)

    # 4. Create time series sequences (sliding windows)
    X, y = create_sequences(scaled_data, window_size=60)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 5. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 6. Reshape for LSTM input: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 7. Build and train the LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    model, history = train_lstm_model(model, X_train, y_train, epochs=20)

    # 8. Save model (optional)
    model.save("models/lstm_crypto_model.h5")

    # 9. Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 10. Predict and compare to actual prices
    predicted_prices = predict_and_inverse(model, X_test, scaler)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 11. Plot actual vs. predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_rescaled, label="Actual Price")
    plt.plot(predicted_prices, label="Predicted Price")
    plt.title("Predicted vs. Actual Crypto Prices")
    plt.xlabel("Time Steps (Test Data)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

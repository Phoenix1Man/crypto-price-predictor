# 🚀 Crypto Price Predictor with LSTM & Sentiment AI


A powerful, AI-driven web app that predicts cryptocurrency prices using **LSTM neural networks** and **real-time news sentiment analysis**, all visualized on an interactive and modern dashboard.

---

## 📌 Features

### 🧠 Machine Learning
- **LSTM Time Series Forecasting**: Predicts future prices using historical hourly data
- **Multivariate Inputs**: Combines price + sentiment for smarter forecasting

### 💬 Sentiment Analysis
- Pulls real crypto headlines via CryptoPanic API
- Uses VADER NLP model to assess headline sentiment
- Injects sentiment scores into LSTM predictions

### 📊 Interactive Dashboard
- Built with **Flask + Plotly** for fast, dynamic chart rendering
- Dark/Light mode toggle
- Multi-tab layout: Bitcoin, Ethereum, BinanceCoin ...


### 📰 Live News Feed
- Displays the top latest crypto news headlines at the bottom of the page

---

## 🛠 Tech Stack

| Component          | Tech             |
|-------------------|------------------|
| Backend            | Python, Flask    |
| ML Framework       | TensorFlow (LSTM)|
| Data Processing    | Pandas, NumPy    |
| Visualization      | Plotly.js        |
| Sentiment Analysis | VADER (NLTK)     |
| Styling/UI         | HTML, CSS        |
| Crypto Data        | CoinGecko API    |
| News Data          | CryptoPanic API  |

---

## 📦 Project Structure

```
crypto_price_predictor/
├── dashboard/
│   ├── app.py                  # Flask app entry point
│   ├── templates/
│   │   └── index.html          # Dashboard UI
├── models/
│   └── lstm_model.py           # LSTM model build + predict
├── utils/
│   ├── data_fetcher.py         # API data collectors
│   └── preprocessing.py        # Data scaler, sequencer, etc.
├── sentiment/
│   └── analyzer.py             # News + VADER integration
├── static/                     # Custom CSS or assets
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/Phoenix1Man/crypto-price-predictor.git
cd crypto-price-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Flask app**
```bash
cd dashboard
python app.py
```

4. **Visit your dashboard**  
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---


## 👨‍💻 Author
**Built by [@Mohamed_Dhia_Jebri](https://github.com/Phoenix1Man)**

---

## 📜 License
MIT License. Free for personal and commercial use.

"# crypto-price-predictor" 

# utils/data_fetcher.py

import requests
import pandas as pd

def get_historical_price(coin_id="bitcoin", vs_currency="usd", days=7):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days
        # Removed interval to avoid 401 error
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        prices = response.json()["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    else:
        print(response.text)
        raise Exception("Failed to fetch data:", response.status_code)

def get_available_coins():
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            coins = response.json()
            # Sort coins by name for better organization
            return sorted(coins, key=lambda x: x['name'])
        else:
            print(f"Failed to fetch coins: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching coins: {e}")
        return []

# sentiment/analyzer.py
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
load_dotenv()
TOKEN = os.getenv("CRYPTOPANIC_TOKEN")


import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_news_headlines():
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={TOKEN}"


    try:
        response = requests.get(url)
        data = response.json()

        if 'results' not in data:
            print("[CryptoPanic] No 'results' key in response:", data)
            return []  # Return empty list if API is limited or failed

        headlines = [(item['title'], item['published_at']) for item in data['results']]
        return headlines[:20]

    except Exception as e:
        print("[CryptoPanic] Error fetching news:", e)
        return []  # Return empty list on network/API errors

def analyze_sentiment(headline_list):
    scores = []
    for text in headline_list:
        sentiment = analyzer.polarity_scores(text)
        scores.append(sentiment['compound'])  # -1 (neg) to +1 (pos)
    return scores

def get_daily_sentiment_map():
    headlines = get_news_headlines()
    if not headlines:
        print("[Sentiment] No headlines available, using neutral sentiment.")
        return {}

    sentiment_by_day = {}

    for title, timestamp in headlines:
        day = timestamp.split("T")[0]
        score = analyzer.polarity_scores(title)['compound']
        sentiment_by_day.setdefault(day, []).append(score)

    sentiment_map = {}
    for day, scores in sentiment_by_day.items():
        sentiment_map[day] = sum(scores) / len(scores)

    return sentiment_map


"""News sentiment analysis."""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

analyzer = SentimentIntensityAnalyzer()


def fetch_news(symbol: str) -> List[dict]:
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        return news
    except Exception as exc:
        logger.debug("No news for %s: %s", symbol, exc)
        return []


def score_sentiment(news_items: List[dict]) -> float:
    if not news_items:
        return 0.0
    scores = []
    for item in news_items[:10]:
        text = " ".join(filter(None, [item.get("title"), item.get("summary")]))
        if not text:
            continue
        scores.append(analyzer.polarity_scores(text)["compound"])
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def sentiment_for_symbol(symbol: str) -> Dict[str, float]:
    news = fetch_news(symbol)
    score = score_sentiment(news)
    return {"sentiment_score": score, "news_count": len(news)}

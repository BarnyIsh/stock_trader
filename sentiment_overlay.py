"""
sentiment_overlay.py - News and social attention overlay for live stock scores.

This is intentionally a capped adjustment layer, not a replacement for the ML
model. It uses recent public Reddit JSON listings and finance/news RSS feeds,
weights more reputable sources higher, and rewards unusual attention volume.
"""

from __future__ import annotations

import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from html import unescape
from urllib.parse import quote_plus
from requests.auth import HTTPBasicAuth

import pandas as pd
import requests


REDDIT_SUBREDDITS = os.getenv(
    "REDDIT_SUBREDDITS",
    "wallstreetbets+stocks+investing+StockMarket",
)
REDDIT_USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    "windows:stock-trader-market-open:v1.0 (by /u/BarnyIsh)",
)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "").strip()
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
REDDIT_LISTINGS = [
    item.strip()
    for item in os.getenv("REDDIT_LISTINGS", "new,hot,rising,top").split(",")
    if item.strip()
]
REDDIT_TOP_TIME_FILTER = os.getenv("REDDIT_TOP_TIME_FILTER", "day")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()
X_TOP_N = int(os.getenv("X_TOP_N", "12"))
SENTIMENT_MAX_ADJUST = float(os.getenv("SENTIMENT_MAX_ADJUST", "0.12"))
NEWS_TOP_N = int(os.getenv("NEWS_TOP_N", "12"))
REQUEST_TIMEOUT = float(os.getenv("SENTIMENT_REQUEST_TIMEOUT", "6"))


POSITIVE_WORDS = {
    "beat", "beats", "bullish", "upgrade", "upgraded", "raises", "raised",
    "growth", "surge", "surges", "rally", "rallies", "record", "strong",
    "outperform", "buy", "profit", "profits", "approval", "approved",
    "partnership", "deal", "guidance", "momentum", "breakout",
}
NEGATIVE_WORDS = {
    "miss", "misses", "bearish", "downgrade", "downgraded", "cuts", "cut",
    "lawsuit", "probe", "investigation", "weak", "warning", "loss",
    "losses", "recall", "fraud", "bankruptcy", "layoff", "layoffs",
    "slump", "plunge", "falls", "sell", "underperform", "risk", "risks",
}
SOURCE_WEIGHTS = {
    "reuters": 1.45,
    "bloomberg": 1.40,
    "associated press": 1.35,
    "ap": 1.35,
    "wall street journal": 1.35,
    "wsj": 1.35,
    "cnbc": 1.25,
    "marketwatch": 1.15,
    "barron's": 1.15,
    "financial times": 1.30,
    "ft": 1.30,
    "yahoo finance": 1.00,
    "seeking alpha": 0.85,
    "benzinga": 0.85,
    "motley fool": 0.65,
    "reddit": 0.55,
    "x": 0.70,
    "twitter": 0.70,
}


@dataclass
class MentionStats:
    ticker: str
    reddit_mentions: int = 0
    reddit_comments: int = 0
    reddit_score: int = 0
    reddit_sentiment: float = 0.0
    x_mentions: int = 0
    x_likes: int = 0
    x_retweets: int = 0
    x_sentiment: float = 0.0
    news_mentions: int = 0
    news_sentiment: float = 0.0
    weighted_news_sentiment: float = 0.0
    attention_score: float = 0.0
    sentiment_adjustment: float = 0.0
    adjusted_prob_buy: float = 0.0


def _sentiment_score(text: str) -> float:
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for word in words if word in POSITIVE_WORDS)
    neg = sum(1 for word in words if word in NEGATIVE_WORDS)
    raw = (pos - neg) / math.sqrt(len(words))
    return max(-1.0, min(1.0, raw))


def _source_weight(source: str) -> float:
    src = source.lower()
    for key, weight in SOURCE_WEIGHTS.items():
        if key in src:
            return weight
    return 0.75


def _ticker_pattern(tickers: list[str]) -> re.Pattern:
    escaped = sorted((re.escape(t) for t in tickers), key=len, reverse=True)
    return re.compile(r"(?<![A-Z$])\$?(" + "|".join(escaped) + r")(?![A-Z])")


def _dedupe_posts(posts: list[dict]) -> list[dict]:
    seen = set()
    rows = []
    for post in posts:
        key = post.get("id") or post.get("permalink") or post.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(post)
    return rows


def _fetch_reddit_listing(
    base_url: str,
    listing: str,
    headers: dict,
    params: dict | None = None,
) -> list[dict]:
    if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
        url = f"{base_url}/{listing}"
    else:
        url = f"{base_url}/{listing}.json"

    try:
        resp = requests.get(
            url,
            params=params or {"limit": 75},
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
        return [child.get("data", {}) for child in children]
    except Exception as exc:
        print(f"[sentiment] Reddit {listing} fetch failed: {exc}")
        return []


def _fetch_reddit_posts() -> list[dict]:
    headers = {"User-Agent": REDDIT_USER_AGENT}
    posts = []
    if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
        try:
            token_resp = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
                data={"grant_type": "client_credentials"},
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            token_resp.raise_for_status()
            token = token_resp.json()["access_token"]
            api_headers = {
                "User-Agent": REDDIT_USER_AGENT,
                "Authorization": f"Bearer {token}",
            }
            base_url = f"https://oauth.reddit.com/r/{REDDIT_SUBREDDITS}"
            for listing in REDDIT_LISTINGS:
                params = {"limit": 75}
                if listing == "top":
                    params["t"] = REDDIT_TOP_TIME_FILTER
                posts.extend(_fetch_reddit_listing(base_url, listing, api_headers, params))
                time.sleep(0.05)
            return _dedupe_posts(posts)
        except Exception as exc:
            print(f"[sentiment] Reddit OAuth fetch failed: {exc}")
            return []

    print(
        "[sentiment] Reddit OAuth credentials not set; attempting public "
        "JSON listings, which Reddit may block."
    )
    base_url = f"https://www.reddit.com/r/{REDDIT_SUBREDDITS}"
    for listing in REDDIT_LISTINGS:
        params = {"limit": 75}
        if listing == "top":
            params["t"] = REDDIT_TOP_TIME_FILTER
        posts.extend(_fetch_reddit_listing(base_url, listing, headers, params))
        time.sleep(0.05)
    return _dedupe_posts(posts)


def _fetch_x_posts(tickers: list[str]) -> list[dict]:
    if not X_BEARER_TOKEN:
        return []
    selected = tickers[:X_TOP_N]
    cashtags = " OR ".join(f"${ticker}" for ticker in selected)
    query = f"({cashtags}) (stock OR stocks OR earnings OR shares OR market) -is:retweet lang:en"
    try:
        resp = requests.get(
            "https://api.x.com/2/tweets/search/recent",
            params={
                "query": query,
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics,lang",
            },
            headers={"Authorization": f"Bearer {X_BEARER_TOKEN}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as exc:
        print(f"[sentiment] X recent search failed: {exc}")
        return []


def _parse_rss(url: str) -> list[dict]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": REDDIT_USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:
        print(f"[sentiment] RSS fetch failed: {url}: {exc}")
        return []

    rows = []
    for item in root.findall(".//item")[:8]:
        title = unescape(item.findtext("title", default=""))
        description = unescape(item.findtext("description", default=""))
        source = item.findtext("{http://www.w3.org/2005/Atom}source", default="")
        if not source:
            source = item.findtext("source", default="")
        rows.append({
            "title": re.sub(r"<[^>]+>", " ", title),
            "description": re.sub(r"<[^>]+>", " ", description),
            "source": source or "news",
        })
    return rows


def _fetch_news_items(ticker: str) -> list[dict]:
    query = quote_plus(f"{ticker} stock")
    urls = [
        f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
    ]
    items = []
    for url in urls:
        items.extend(_parse_rss(url))
        time.sleep(0.05)
    return items[:10]


def _fetch_market_news_items() -> list[dict]:
    query = quote_plus("stock market earnings shares company")
    return _parse_rss(
        f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    )[:50]


def build_sentiment_overlay(
    scored_df: pd.DataFrame,
    tickers: list[str],
    top_n_news: int = NEWS_TOP_N,
) -> pd.DataFrame:
    """Return one row per ticker with attention and score adjustment columns."""
    if scored_df.empty:
        return pd.DataFrame()

    tickers = list(dict.fromkeys(tickers))
    stats = {ticker: MentionStats(ticker=ticker) for ticker in tickers}
    pattern = _ticker_pattern(tickers)

    for post in _fetch_reddit_posts():
        title = post.get("title", "")
        body = post.get("selftext", "")
        text = f"{title} {body}"
        mentioned = set(pattern.findall(text.upper()))
        if not mentioned:
            continue
        sentiment = _sentiment_score(text)
        comments = int(post.get("num_comments", 0) or 0)
        score = int(post.get("score", 0) or 0)
        attention = 1 + math.log1p(max(comments, 0)) + 0.25 * math.log1p(max(score, 0))
        for ticker in mentioned:
            item = stats.get(ticker)
            if item is None:
                continue
            item.reddit_mentions += 1
            item.reddit_comments += comments
            item.reddit_score += score
            item.reddit_sentiment += sentiment * attention

    for item in stats.values():
        if item.reddit_mentions:
            denom = item.reddit_mentions + math.log1p(max(item.reddit_comments, 0))
            item.reddit_sentiment = item.reddit_sentiment / max(denom, 1)

    ranked_tickers = (
        scored_df.sort_values("prob_buy", ascending=False)["ticker"]
        .head(max(X_TOP_N, NEWS_TOP_N))
        .tolist()
    )
    for post in _fetch_x_posts(ranked_tickers):
        text = post.get("text", "")
        mentioned = set(pattern.findall(text.upper()))
        if not mentioned:
            continue
        metrics = post.get("public_metrics", {}) or {}
        likes = int(metrics.get("like_count", 0) or 0)
        retweets = int(metrics.get("retweet_count", 0) or 0)
        replies = int(metrics.get("reply_count", 0) or 0)
        sentiment = _sentiment_score(text)
        attention = (
            1
            + math.log1p(max(likes, 0))
            + math.log1p(max(retweets, 0))
            + 0.5 * math.log1p(max(replies, 0))
        )
        for ticker in mentioned:
            item = stats.get(ticker)
            if item is None:
                continue
            item.x_mentions += 1
            item.x_likes += likes
            item.x_retweets += retweets
            item.x_sentiment += sentiment * attention

    for item in stats.values():
        if item.x_mentions:
            denom = (
                item.x_mentions
                + math.log1p(max(item.x_likes, 0))
                + math.log1p(max(item.x_retweets, 0))
            )
            item.x_sentiment = item.x_sentiment / max(denom, 1)

    for news in _fetch_market_news_items():
        text = f"{news['title']} {news['description']}"
        mentioned = set(pattern.findall(text.upper()))
        if not mentioned:
            continue
        sentiment = _sentiment_score(text)
        weight = _source_weight(news.get("source", ""))
        for ticker in mentioned:
            item = stats.get(ticker)
            if item is None:
                continue
            current_total = item.weighted_news_sentiment * max(item.news_mentions, 1)
            item.news_mentions += 1
            item.weighted_news_sentiment = (
                current_total + sentiment * weight
            ) / max(item.news_mentions, 1)

    news_tickers = (
        scored_df.sort_values("prob_buy", ascending=False)["ticker"]
        .head(top_n_news)
        .tolist()
    )
    for ticker in news_tickers:
        item = stats.get(ticker)
        if item is None:
            continue
        weighted_sum = 0.0
        total_weight = 0.0
        for news in _fetch_news_items(ticker):
            text = f"{news['title']} {news['description']}"
            sentiment = _sentiment_score(text)
            weight = _source_weight(news.get("source", ""))
            weighted_sum += sentiment * weight
            total_weight += weight
            if re.search(rf"(?<![A-Z]){re.escape(ticker)}(?![A-Z])", text.upper()):
                item.news_mentions += 1
        if total_weight:
            targeted_sentiment = weighted_sum / total_weight
            if item.weighted_news_sentiment:
                item.weighted_news_sentiment = (
                    0.55 * item.weighted_news_sentiment
                    + 0.45 * targeted_sentiment
                )
            else:
                item.weighted_news_sentiment = targeted_sentiment
            item.news_sentiment = weighted_sum / max(item.news_mentions, 1)

    mention_counts = [
        s.reddit_mentions
        + s.x_mentions
        + s.news_mentions
        + math.log1p(max(s.reddit_comments, 0))
        + math.log1p(max(s.x_likes + s.x_retweets, 0))
        for s in stats.values()
    ]
    mean_attention = sum(mention_counts) / max(len(mention_counts), 1)
    variance = sum((x - mean_attention) ** 2 for x in mention_counts) / max(len(mention_counts), 1)
    std_attention = math.sqrt(variance) or 1.0

    base_probs = dict(zip(scored_df["ticker"], scored_df["prob_buy"]))
    for ticker, item in stats.items():
        count = (
            item.reddit_mentions
            + item.x_mentions
            + item.news_mentions
            + math.log1p(max(item.reddit_comments, 0))
            + math.log1p(max(item.x_likes + item.x_retweets, 0))
        )
        z_attention = max(0.0, min(2.0, (count - mean_attention) / std_attention))
        item.attention_score = z_attention / 2.0

        sentiment = (
            0.55 * item.weighted_news_sentiment
            + 0.18 * item.reddit_sentiment
            + 0.12 * item.x_sentiment
            + 0.20 * item.attention_score
        )
        item.sentiment_adjustment = max(
            -SENTIMENT_MAX_ADJUST,
            min(SENTIMENT_MAX_ADJUST, sentiment * SENTIMENT_MAX_ADJUST),
        )
        item.adjusted_prob_buy = float(base_probs.get(ticker, 0.0)) + item.sentiment_adjustment

    return pd.DataFrame([asdict(item) for item in stats.values()])


def apply_sentiment_overlay(scored_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored_df.empty:
        return scored_df, pd.DataFrame()
    overlay = build_sentiment_overlay(scored_df, scored_df["ticker"].tolist())
    if overlay.empty:
        scored_df = scored_df.copy()
        scored_df["base_prob_buy"] = scored_df["prob_buy"]
        scored_df["sentiment_adjustment"] = 0.0
        return scored_df, overlay

    merged = scored_df.merge(
        overlay[[
            "ticker", "sentiment_adjustment", "adjusted_prob_buy",
            "reddit_mentions", "reddit_comments", "reddit_score",
            "reddit_sentiment", "x_mentions", "x_likes", "x_retweets",
            "x_sentiment", "news_mentions", "weighted_news_sentiment",
            "attention_score",
        ]],
        on="ticker",
        how="left",
    )
    merged["base_prob_buy"] = merged["prob_buy"]
    merged["sentiment_adjustment"] = merged["sentiment_adjustment"].fillna(0.0)
    merged["adjusted_prob_buy"] = (
        merged["adjusted_prob_buy"].fillna(merged["base_prob_buy"])
    )
    merged["prob_buy"] = merged["adjusted_prob_buy"]
    merged = merged.sort_values("prob_buy", ascending=False).reset_index(drop=True)
    return merged, overlay.sort_values("adjusted_prob_buy", ascending=False).reset_index(drop=True)

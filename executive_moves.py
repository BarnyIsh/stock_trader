"""
executive_moves.py - Track executive/key personnel movements as a trading signal.

Thesis: When senior executives (CEO, CFO, CTO, etc.) leave a company — especially
to join a competitor — it often signals internal problems or a talent drain that
precedes stock underperformance. Conversely, a company *attracting* top talent
from competitors can signal strength.

Data sources:
  - Google News RSS (free, no auth)
  - Finnhub company news API (if key available)

Features produced per ticker:
  - exec_departures_30d: count of executive departures in last 30 days
  - exec_arrivals_30d: count of executive arrivals in last 30 days
  - exec_net_flow_30d: arrivals - departures (positive = gaining talent)
  - exec_departure_severity: weighted severity (CEO=5, CFO=4, CTO=3, VP=2, Dir=1)
  - exec_move_sentiment: net sentiment of executive movement news (-1 to 1)
"""

from __future__ import annotations

import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import pandas as pd
import requests

from config import EXEC_MOVE_LOOKBACK_DAYS, EXEC_NEWS_TIMEOUT


# ─── Role severity weights ───────────────────────────────────────────────────
# Higher = more market impact when this person leaves/joins

ROLE_WEIGHTS = {
    "ceo": 5.0,
    "chief executive": 5.0,
    "founder": 4.5,
    "co-founder": 4.5,
    "cfo": 4.0,
    "chief financial": 4.0,
    "cto": 3.5,
    "chief technology": 3.5,
    "coo": 3.5,
    "chief operating": 3.5,
    "cio": 3.0,
    "chief information": 3.0,
    "chief ai": 3.5,
    "chief product": 3.0,
    "president": 3.0,
    "general counsel": 2.5,
    "evp": 2.5,
    "executive vice president": 2.5,
    "svp": 2.0,
    "senior vice president": 2.0,
    "vp": 1.5,
    "vice president": 1.5,
    "head of": 1.5,
    "director": 1.0,
    "managing director": 1.5,
    "partner": 1.5,
    "board member": 2.5,
    "chairman": 4.0,
}

# Keywords indicating someone is LEAVING
DEPARTURE_KEYWORDS = {
    "leaves", "left", "departs", "departed", "exits", "exited",
    "steps down", "stepped down", "resign", "resigns", "resigned",
    "retiring", "retired", "fired", "ousted", "replaced",
    "transition out", "stepping aside", "quit", "quits",
    "departure", "exit", "leaving",
}

# Keywords indicating someone is ARRIVING
ARRIVAL_KEYWORDS = {
    "joins", "joined", "hires", "hired", "appoints", "appointed",
    "names", "named", "promotes", "promoted", "welcomes",
    "taps", "recruits", "recruited", "brings on", "new hire",
    "appointment", "joining", "onboards",
}


@dataclass
class ExecMove:
    """Represents a single executive movement event."""
    ticker: str
    direction: str  # "departure" or "arrival"
    role: str
    severity: float
    headline: str
    date: str
    source: str = ""


@dataclass
class ExecMoveScore:
    """Aggregated executive movement features for a single ticker."""
    ticker: str
    exec_departures_30d: int = 0
    exec_arrivals_30d: int = 0
    exec_net_flow_30d: int = 0
    exec_departure_severity: float = 0.0
    exec_move_sentiment: float = 0.0


def _detect_role(text: str) -> tuple[str, float]:
    """Detect the highest-ranking role mentioned in text and return (role, weight)."""
    text_lower = text.lower()
    best_role = ""
    best_weight = 0.0
    for role, weight in ROLE_WEIGHTS.items():
        if role in text_lower and weight > best_weight:
            best_role = role
            best_weight = weight
    return best_role, best_weight


def _detect_direction(text: str) -> str | None:
    """Detect if the headline describes a departure or arrival."""
    text_lower = text.lower()
    dep_count = sum(1 for kw in DEPARTURE_KEYWORDS if kw in text_lower)
    arr_count = sum(1 for kw in ARRIVAL_KEYWORDS if kw in text_lower)
    if dep_count > arr_count:
        return "departure"
    elif arr_count > dep_count:
        return "arrival"
    return None


def _fetch_exec_news_google(ticker: str, company_name: str = "") -> list[dict]:
    """Fetch executive movement news from Google News RSS."""
    search_term = company_name if company_name else ticker
    queries = [
        f"{search_term} CEO leaves OR departs OR joins OR hires OR appoints",
        f"{search_term} executive departure OR appointment OR resigns",
        f"{search_term} CFO OR CTO OR COO leaves OR joins OR hired",
    ]

    articles = []
    seen_titles = set()

    for query in queries:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "StockTrader/1.0"},
                timeout=EXEC_NEWS_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item"):
                title = item.findtext("title", "")
                pub_date = item.findtext("pubDate", "")
                source = item.findtext("source", "")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                articles.append({
                    "title": title,
                    "pub_date": pub_date,
                    "source": source,
                })
        except Exception:
            continue
        time.sleep(0.3)

    return articles


def _fetch_exec_news_finnhub(ticker: str) -> list[dict]:
    """Fetch company news from Finnhub API (if key available)."""
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return []

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=EXEC_MOVE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={start_date}&to={end_date}&token={api_key}"
    )
    try:
        resp = requests.get(url, timeout=EXEC_NEWS_TIMEOUT)
        if resp.status_code != 200:
            return []
        data = resp.json()
        # Filter for executive-related news
        exec_articles = []
        for article in data:
            headline = article.get("headline", "")
            if _detect_role(headline)[1] > 0 and _detect_direction(headline):
                exec_articles.append({
                    "title": headline,
                    "pub_date": datetime.fromtimestamp(
                        article.get("datetime", 0)
                    ).strftime("%a, %d %b %Y %H:%M:%S GMT"),
                    "source": article.get("source", ""),
                })
        return exec_articles
    except Exception:
        return []


def _parse_pub_date(pub_date: str) -> datetime | None:
    """Parse various RSS date formats."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(pub_date.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def _is_within_lookback(pub_date: str, lookback_days: int) -> bool:
    """Check if a publication date is within the lookback window."""
    parsed = _parse_pub_date(pub_date)
    if parsed is None:
        # If we can't parse the date, include it (conservative)
        return True
    cutoff = datetime.now() - timedelta(days=lookback_days)
    # Make both offset-naive for comparison
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed >= cutoff


def detect_exec_moves(ticker: str, company_name: str = "") -> list[ExecMove]:
    """
    Detect executive movements for a given ticker from news sources.
    Returns a list of ExecMove events within the lookback window.
    """
    # Gather articles from available sources
    articles = _fetch_exec_news_google(ticker, company_name)
    articles.extend(_fetch_exec_news_finnhub(ticker))

    moves = []
    for article in articles:
        title = article.get("title", "")
        pub_date = article.get("pub_date", "")

        # Filter by lookback window
        if not _is_within_lookback(pub_date, EXEC_MOVE_LOOKBACK_DAYS):
            continue

        # Detect role and direction
        role, severity = _detect_role(title)
        if severity == 0:
            continue  # No executive role detected

        direction = _detect_direction(title)
        if direction is None:
            continue  # Can't determine if departure or arrival

        moves.append(ExecMove(
            ticker=ticker,
            direction=direction,
            role=role,
            severity=severity,
            headline=title,
            date=pub_date,
            source=article.get("source", ""),
        ))

    return moves


def score_exec_moves(ticker: str, company_name: str = "") -> ExecMoveScore:
    """
    Compute aggregated executive movement features for a ticker.
    This is the main entry point called by the feature pipeline.
    """
    moves = detect_exec_moves(ticker, company_name)
    score = ExecMoveScore(ticker=ticker)

    if not moves:
        return score

    departures = [m for m in moves if m.direction == "departure"]
    arrivals = [m for m in moves if m.direction == "arrival"]

    score.exec_departures_30d = len(departures)
    score.exec_arrivals_30d = len(arrivals)
    score.exec_net_flow_30d = len(arrivals) - len(departures)

    # Severity: sum of departure severities (higher = more talent loss)
    score.exec_departure_severity = sum(m.severity for m in departures)

    # Sentiment: normalized net flow weighted by severity
    total_arrival_weight = sum(m.severity for m in arrivals)
    total_departure_weight = sum(m.severity for m in departures)
    total_weight = total_arrival_weight + total_departure_weight
    if total_weight > 0:
        # Range: -1 (all departures) to +1 (all arrivals)
        score.exec_move_sentiment = (
            (total_arrival_weight - total_departure_weight) / total_weight
        )

    return score


def build_exec_move_features(tickers: list[str]) -> pd.DataFrame:
    """
    Build executive movement feature DataFrame for a list of tickers.
    Returns DataFrame with ticker as index and exec_* columns.

    Used by features.py build_feature_matrix() for live scoring,
    and can be joined to historical feature matrices.
    """
    rows = []
    for ticker in tickers:
        score = score_exec_moves(ticker)
        rows.append({
            "ticker": ticker,
            "exec_departures_30d": score.exec_departures_30d,
            "exec_arrivals_30d": score.exec_arrivals_30d,
            "exec_net_flow_30d": score.exec_net_flow_30d,
            "exec_departure_severity": score.exec_departure_severity,
            "exec_move_sentiment": score.exec_move_sentiment,
        })
        time.sleep(0.2)  # rate limit for news API

    return pd.DataFrame(rows)

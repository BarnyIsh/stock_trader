"""
sentiment_overlay.py - News and social attention overlay for live stock scores.

This is intentionally a capped adjustment layer, not a replacement for the ML
model. It uses recent public Reddit JSON listings and finance/news RSS feeds,
weights more reputable sources higher, and rewards unusual attention volume.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from html import unescape
from urllib.parse import quote_plus

import pandas as pd
import requests


REDDIT_SUBREDDITS = os.getenv(
    "REDDIT_SUBREDDITS",
    "wallstreetbets+investing+stocks+news",
)
REDDIT_USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
)
REDDIT_COOKIE = os.getenv("REDDIT_COOKIE", "").strip()
REDDIT_LISTINGS = [
    item.strip()
    for item in os.getenv("REDDIT_LISTINGS", "hot,new,rising,top").split(",")
    if item.strip()
]
REDDIT_TOP_TIME_FILTER = os.getenv("REDDIT_TOP_TIME_FILTER", "day")
X_TOP_N = int(os.getenv("X_TOP_N", "12"))
X_SEARCH_PAGES = int(os.getenv("X_SEARCH_PAGES", "2"))
X_AUTH_TOKEN = os.getenv("X_AUTH_TOKEN", "").strip()
X_CT0 = os.getenv("X_CT0", "").strip()
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()
X_RUNTIME_BROWSER_INSTALL = os.getenv("X_RUNTIME_BROWSER_INSTALL", "false").lower() == "true"
PLAYWRIGHT_CDP_URL = os.getenv("PLAYWRIGHT_CDP_URL", "").strip()
PLAYWRIGHT_WS_ENDPOINT = os.getenv("PLAYWRIGHT_WS_ENDPOINT", "").strip()
SENTIMENT_MAX_ADJUST = float(os.getenv("SENTIMENT_MAX_ADJUST", "0.12"))
NEWS_TOP_N = int(os.getenv("NEWS_TOP_N", "12"))
REQUEST_TIMEOUT = float(os.getenv("SENTIMENT_REQUEST_TIMEOUT", "6"))
SOURCE_STATUS: dict[str, str] = {}
PLAYWRIGHT_BROWSER_PATH = os.getenv(
    "PLAYWRIGHT_BROWSERS_PATH",
    "/tmp/playwright-browsers" if os.getenv("VERCEL") else "",
).strip()


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


def _reset_source_status():
    SOURCE_STATUS.clear()
    SOURCE_STATUS.update({
        "x": "not configured",
        "reddit": "not checked",
        "google_news": "not checked",
    })


def _mark_source(source: str, status: str):
    SOURCE_STATUS[source] = status


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
        post_id = post.get("id", "")
        key = f"{post.get('subreddit', '')}:{post_id}" if post_id else (
            post.get("permalink") or post.get("title")
        )
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(post)
    return rows


def _reddit_subreddit_names() -> list[str]:
    raw = re.split(r"[,+\s]+", REDDIT_SUBREDDITS)
    names = []
    for item in raw:
        name = item.strip().strip("/")
        if not name:
            continue
        if name.lower().startswith("r/"):
            name = name[2:]
        names.append(name)
    return list(dict.fromkeys(names))


def _reddit_headers() -> dict:
    headers = {
        "User-Agent": REDDIT_USER_AGENT,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if REDDIT_COOKIE:
        headers["Cookie"] = REDDIT_COOKIE
    return headers


def _fetch_reddit_listing(
    subreddit: str,
    listing: str,
    headers: dict,
    params: dict | None = None,
) -> tuple[list[dict], str | None]:
    """Fetch Reddit JSON listing via direct HTTP request."""
    if listing == "hot":
        url = f"https://www.reddit.com/r/{subreddit}.json"
    else:
        url = f"https://www.reddit.com/r/{subreddit}/{listing}.json"

    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 403:
            return [], f"r/{subreddit} {listing}: 403 blocked"
        resp.raise_for_status()
        data = resp.json()
        children = data.get("data", {}).get("children", [])
        if not children:
            return [], f"r/{subreddit} {listing}: empty response"
        rows = []
        for child in children:
            post = child.get("data", {}) or {}
            post.setdefault("subreddit", subreddit)
            rows.append(post)
        return rows, None
    except Exception as exc:
        error = f"r/{subreddit} {listing}: {exc}"
        print(f"[sentiment] Reddit {error}")
        return [], error


def _fetch_reddit_listing_with_playwright(
    subreddit: str,
    listing: str,
    params: dict | None = None,
) -> tuple[list[dict], str | None]:
    """Fetch Reddit JSON listing via a Playwright browser (remote or local).
    NOTE: This is the single-call version kept for backward compat.
    Prefer _fetch_all_browser_sources for production use.
    """
    results = _fetch_reddit_bulk_with_playwright([(subreddit, listing, params)])
    if results:
        return results[0]
    return [], "no results from bulk fetch"


def _reddit_page_fetch(page, subreddit: str, listing: str, params: dict | None) -> tuple[list[dict], str | None]:
    """Fetch a single Reddit JSON listing using an existing browser page."""
    if listing == "hot":
        url = f"https://www.reddit.com/r/{subreddit}.json"
    else:
        url = f"https://www.reddit.com/r/{subreddit}/{listing}.json"
    if params:
        query = "&".join(f"{key}={quote_plus(str(value))}" for key, value in params.items())
        url = f"{url}?{query}"

    try:
        json_response = {}

        def handle_response(response):
            nonlocal json_response
            if response.url.startswith(url.split("?")[0]) and response.status == 200:
                try:
                    json_response = response.json()
                except Exception:
                    pass

        page.on("response", handle_response)
        page.goto(url, wait_until="networkidle", timeout=int(REQUEST_TIMEOUT * 2000))
        page.remove_listener("response", handle_response)

        if json_response and "data" in json_response:
            payload = json_response
        else:
            text = page.locator("body").inner_text(timeout=int(REQUEST_TIMEOUT * 1000)).strip()
            if text.startswith("{"):
                payload = json.loads(text)
            else:
                pre = page.locator("pre").first
                try:
                    pre_text = pre.inner_text(timeout=2000).strip()
                    payload = json.loads(pre_text)
                except Exception:
                    match = re.search(r"(\{\"kind\".*\})\s*$", text, flags=re.DOTALL)
                    if not match:
                        return [], f"non-json from r/{subreddit}/{listing}"
                    payload = json.loads(match.group(1))

        rows = []
        for child in payload.get("data", {}).get("children", []):
            post = child.get("data", {}) or {}
            post.setdefault("subreddit", subreddit)
            rows.append(post)
        return rows, None
    except Exception as exc:
        return [], f"r/{subreddit}/{listing}: {exc}"


def _x_page_scrape(context, search_urls: list[str]) -> list[dict]:
    """Scrape X search results using an existing browser context."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    _add_x_session_cookies(context)
    page = context.new_page()

    posts: list[dict] = []
    seen: set[str] = set()
    login_required = False
    empty_reason = ""

    for search_url in search_urls:
        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=int(REQUEST_TIMEOUT * 1000))
        except Exception:
            continue

        for _ in range(max(X_SEARCH_PAGES, 1)):
            try:
                page.wait_for_selector("article[data-testid='tweet']", timeout=3500)
            except (PlaywrightTimeoutError, Exception):
                empty_reason = _x_empty_reason(page)
                break

            articles = page.locator("article[data-testid='tweet']").all()
            for article in articles:
                text = _x_article_text(article)
                if not text:
                    continue
                key = re.sub(r"\s+", " ", text).strip()[:240]
                if key in seen:
                    continue
                seen.add(key)
                posts.append({
                    "text": text,
                    "public_metrics": _x_article_metrics(article),
                })
                if len(posts) >= 100:
                    break
            if len(posts) >= 100:
                break
            try:
                page.mouse.wheel(0, 1600)
                page.wait_for_timeout(900)
            except Exception:
                break
        if len(posts) >= 100:
            break
        if _x_login_required(page):
            login_required = True
            break

    page.close()

    if posts:
        _mark_source("x", f"ok: scraped {len(posts)} posts")
    elif login_required:
        _mark_source("x", "unavailable: login required; set X_AUTH_TOKEN and X_CT0")
    elif empty_reason:
        _mark_source("x", f"unavailable: no posts rendered ({empty_reason})")
    else:
        _mark_source("x", "ok: no public posts found")
    return posts


def _fetch_all_browser_sources(
    reddit_requests: list[tuple[str, str, dict | None]],
    x_search_urls: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    Open ONE browser connection and fetch both Reddit JSON and X posts.
    Returns (reddit_posts, x_posts).
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        _mark_source("reddit", f"unavailable: Playwright import failed: {exc}")
        _mark_source("x", f"unavailable: Playwright import failed: {exc}")
        return [], []

    reddit_posts: list[dict] = []
    x_posts: list[dict] = []

    try:
        with sync_playwright() as p:
            browser = _open_playwright_browser(p)

            # --- Reddit: fetch JSON listings ---
            if reddit_requests:
                reddit_context = browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    user_agent=REDDIT_USER_AGENT,
                )
                _add_cookie_header_cookies(reddit_context, REDDIT_COOKIE, ".reddit.com")
                reddit_page = reddit_context.new_page()

                reddit_errors = []
                for subreddit, listing, params in reddit_requests:
                    rows, error = _reddit_page_fetch(reddit_page, subreddit, listing, params)
                    reddit_posts.extend(rows)
                    if error:
                        reddit_errors.append(error)
                    time.sleep(0.3)

                reddit_page.close()
                reddit_context.close()

                deduped = _dedupe_posts(reddit_posts)
                reddit_posts = deduped
                if reddit_posts:
                    _mark_source("reddit", f"ok: fetched {len(reddit_posts)} posts via remote browser")
                else:
                    sample = "; ".join(reddit_errors[:2])
                    _mark_source("reddit", f"unavailable: remote browser ({sample})")

            # --- X: scrape search pages ---
            if x_search_urls:
                x_context = browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0 Safari/537.36"
                    ),
                )
                x_posts = _x_page_scrape(x_context, x_search_urls)
                x_context.close()

            browser.close()

    except Exception as exc:
        error_msg = str(exc)
        if "429" in error_msg:
            _mark_source("reddit", f"unavailable: Browserless 429 rate limit")
            _mark_source("x", f"unavailable: Browserless 429 rate limit")
        else:
            if not reddit_posts:
                _mark_source("reddit", f"unavailable: browser failed: {error_msg[:120]}")
            if not x_posts:
                _mark_source("x", f"unavailable: browser failed: {error_msg[:120]}")

    return reddit_posts, x_posts


def _fetch_reddit_bulk_with_playwright(
    requests_list: list[tuple[str, str, dict | None]],
) -> list[tuple[list[dict], str | None]]:
    """Backward-compat wrapper: fetch Reddit via shared browser (no X)."""
    reddit_posts, _ = _fetch_all_browser_sources(requests_list, [])
    # Return as single result tuple for compat
    if reddit_posts:
        return [(reddit_posts, None)]
    return [([], "browser fetch failed")]


def _fetch_reddit_posts() -> list[dict]:
    """Fetch Reddit posts using multiple strategies in priority order:
    1. Direct JSON (works from some IPs)
    2. RSS feeds (no auth needed, works from most IPs)
    3. Remote browser via Playwright (last resort)
    """
    subreddits = _reddit_subreddit_names()
    if not subreddits:
        _mark_source("reddit", "not configured")
        return []

    # Strategy 1: Try direct JSON fetch
    headers = _reddit_headers()
    test_rows, test_error = _fetch_reddit_listing(subreddits[0], "hot", headers)

    if test_rows:
        # Direct JSON works — use it for everything
        posts = list(test_rows)
        for subreddit in subreddits:
            for listing in REDDIT_LISTINGS:
                if subreddit == subreddits[0] and listing == "hot":
                    continue
                params = None if listing == "hot" else {"limit": 75}
                if listing == "top" and params is not None:
                    params["t"] = REDDIT_TOP_TIME_FILTER
                rows, _ = _fetch_reddit_listing(subreddit, listing, headers, params)
                posts.extend(rows)
                time.sleep(0.05)
        deduped = _dedupe_posts(posts)
        if deduped:
            _mark_source("reddit", f"ok: fetched {len(deduped)} posts via JSON")
            return deduped

    # Strategy 2: RSS feeds (works from most IPs, no auth needed)
    rss_posts = _fetch_reddit_rss(subreddits)
    if rss_posts:
        _mark_source("reddit", f"ok: fetched {len(rss_posts)} posts via RSS")
        return rss_posts

    # Strategy 3: Signal that browser fallback is needed
    # Return None so build_sentiment_overlay uses the shared browser session
    return None  # type: ignore


def _fetch_reddit_rss(subreddits: list[str]) -> list[dict]:
    """Fetch Reddit posts via RSS feeds. No auth, no browser needed."""
    posts = []
    for subreddit in subreddits:
        for listing in REDDIT_LISTINGS:
            if listing == "hot":
                url = f"https://www.reddit.com/r/{subreddit}/.rss"
            elif listing == "top":
                url = f"https://www.reddit.com/r/{subreddit}/top/.rss?t={REDDIT_TOP_TIME_FILTER}"
            else:
                url = f"https://www.reddit.com/r/{subreddit}/{listing}/.rss"

            try:
                resp = requests.get(
                    url,
                    headers={
                        "User-Agent": REDDIT_USER_AGENT,
                        "Accept": "application/rss+xml,application/xml,text/xml,*/*",
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 403:
                    continue
                if resp.status_code == 429:
                    # Rate limited — wait and try next subreddit
                    time.sleep(2.0)
                    continue
                resp.raise_for_status()

                root = ET.fromstring(resp.content)
                # Atom feed namespace
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                for entry in root.findall(".//atom:entry", ns):
                    title = entry.findtext("atom:title", default="", namespaces=ns)
                    content = entry.findtext("atom:content", default="", namespaces=ns)
                    # Strip HTML from content
                    clean_content = re.sub(r"<[^>]+>", " ", content)
                    # Extract subreddit from category
                    category = entry.find("atom:category", ns)
                    sub = category.get("label", subreddit) if category is not None else subreddit

                    posts.append({
                        "title": title,
                        "selftext": clean_content[:2000],
                        "subreddit": sub,
                        "num_comments": 0,
                        "score": 0,
                    })
            except Exception as exc:
                print(f"[sentiment] Reddit RSS r/{subreddit}/{listing} failed: {exc}")
                continue
            # Delay between requests to avoid rate limiting
            time.sleep(1.0)

    return _dedupe_posts(posts) if posts else []


def _build_reddit_browser_requests() -> list[tuple[str, str, dict | None]]:
    """Build the list of Reddit requests for the shared browser session."""
    subreddits = _reddit_subreddit_names()
    requests_list = []
    for subreddit in subreddits:
        for listing in REDDIT_LISTINGS:
            params = None if listing == "hot" else {"limit": 75}
            if listing == "top" and params is not None:
                params["t"] = REDDIT_TOP_TIME_FILTER
            requests_list.append((subreddit, listing, params))
    return requests_list


def _fetch_x_posts_api(tickers: list[str]) -> tuple[list[dict], bool]:
    """
    Fetch recent X/Twitter posts using the v2 API with Bearer token.
    Returns (posts, success). If the API call fails or token is missing,
    returns ([], False) so the caller can fall back to other methods.
    """
    if not X_BEARER_TOKEN:
        return [], False

    selected = tickers[:X_TOP_N]
    if not selected:
        return [], True

    headers = {
        "Authorization": f"Bearer {X_BEARER_TOKEN}",
        "User-Agent": "StockTrader/1.0",
    }
    posts: list[dict] = []

    for ticker in selected:
        query = f"${ticker} (stock OR earnings OR shares OR market) lang:en -is:retweet"
        params = {
            "query": query,
            "max_results": 10,
            "tweet.fields": "public_metrics,text,created_at",
        }
        try:
            resp = requests.get(
                "https://api.x.com/2/tweets/search/recent",
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 401:
                print(f"[sentiment] X API 401: bearer token invalid")
                _mark_source("x", "unavailable: X_BEARER_TOKEN is invalid (401)")
                return posts, bool(posts)
            if resp.status_code == 403:
                print(f"[sentiment] X API 403: insufficient permissions (need Basic tier+)")
                _mark_source("x", "unavailable: X API 403; upgrade to Basic tier for search")
                return posts, bool(posts)
            if resp.status_code == 429:
                print(f"[sentiment] X API rate limited")
                _mark_source("x", f"ok: fetched {len(posts)} posts via API (rate limited)")
                return posts, bool(posts)
            resp.raise_for_status()
            data = resp.json()
            for tweet in data.get("data", []):
                posts.append({
                    "text": tweet.get("text", ""),
                    "public_metrics": tweet.get("public_metrics", {}),
                })
        except Exception as exc:
            print(f"[sentiment] X API error for ${ticker}: {exc}")
            if not posts:
                return [], False
            break
        time.sleep(0.2)

    if posts:
        _mark_source("x", f"ok: fetched {len(posts)} posts via API")
    return posts, True


def _fetch_x_posts(tickers: list[str]) -> list[dict]:
    """Fetch X posts via API. If API unavailable, try Playwright as standalone."""
    selected = tickers[:X_TOP_N]
    if not selected:
        _mark_source("x", "no tickers")
        return []

    # Try X API v2 with Bearer token
    posts, success = _fetch_x_posts_api(selected)
    if success:
        return posts

    # API failed — try Playwright standalone (single browser connection)
    if PLAYWRIGHT_CDP_URL or PLAYWRIGHT_WS_ENDPOINT:
        try:
            x_search_urls = [_x_search_url(t) for t in selected]
            _, x_posts = _fetch_all_browser_sources([], x_search_urls)
            if x_posts:
                return posts + x_posts
        except Exception as exc:
            print(f"[sentiment] X Playwright fallback failed: {exc}")

    if not posts:
        if not SOURCE_STATUS.get("x", "").startswith(("ok", "unavailable")):
            _mark_source("x", "unavailable: API failed, no browser available")
    return posts


def _x_search_url(ticker: str) -> str:
    query = f"${ticker} (stock OR stocks OR earnings OR shares OR market) lang:en"
    return f"https://x.com/search?q={quote_plus(query)}&src=typed_query&f=live"


def _scrape_x_with_playwright(sync_playwright, timeout_error, search_urls: list[str]) -> list[dict]:
    with sync_playwright() as p:
        browser = _open_playwright_browser(p)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
        )
        _add_x_session_cookies(context)
        page = context.new_page()

        posts: list[dict] = []
        seen: set[str] = set()
        login_required = False
        empty_reason = ""
        for search_url in search_urls:
            page.goto(search_url, wait_until="domcontentloaded", timeout=int(REQUEST_TIMEOUT * 1000))
            for _ in range(max(X_SEARCH_PAGES, 1)):
                try:
                    page.wait_for_selector("article[data-testid='tweet']", timeout=3500)
                except timeout_error:
                    empty_reason = _x_empty_reason(page)
                    break

                articles = page.locator("article[data-testid='tweet']").all()
                for article in articles:
                    text = _x_article_text(article)
                    if not text:
                        continue
                    key = re.sub(r"\s+", " ", text).strip()[:240]
                    if key in seen:
                        continue
                    seen.add(key)
                    posts.append({
                        "text": text,
                        "public_metrics": _x_article_metrics(article),
                    })
                    if len(posts) >= 100:
                        break
                if len(posts) >= 100:
                    break
                page.mouse.wheel(0, 1600)
                page.wait_for_timeout(900)
            if len(posts) >= 100:
                break
            if _x_login_required(page):
                login_required = True
                break

        browser.close()
        if posts:
            _mark_source("x", f"ok: scraped {len(posts)} posts")
        elif login_required:
            _mark_source("x", "unavailable: login required; set X_AUTH_TOKEN and X_CT0")
        elif empty_reason:
            _mark_source("x", f"unavailable: no posts rendered ({empty_reason})")
        else:
            _mark_source("x", "ok: no public posts found")
        return posts


def _open_playwright_browser(playwright):
    if PLAYWRIGHT_CDP_URL:
        return playwright.chromium.connect_over_cdp(PLAYWRIGHT_CDP_URL)
    if PLAYWRIGHT_WS_ENDPOINT:
        return playwright.chromium.connect(PLAYWRIGHT_WS_ENDPOINT)
    return playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-sandbox",
        ],
    )


def _playwright_system_lib_missing(exc: Exception) -> bool:
    message = str(exc)
    return (
        "error while loading shared libraries" in message
        or "libnspr4.so" in message
        or "libnss3.so" in message
    )


def _install_playwright_browser() -> bool:
    install_path = "/tmp/playwright-browsers" if os.getenv("VERCEL") else (
        PLAYWRIGHT_BROWSER_PATH or "/tmp/playwright-browsers"
    )
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = install_path
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = install_path

    try:
        _mark_source("x", "installing browser in /tmp")
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--only-shell", "chromium"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        _mark_source("x", "retrying after browser install")
        return True
    except Exception as exc:
        _mark_source("x", f"unavailable: browser install failed: {exc}")
        print(f"[sentiment] Playwright browser install failed: {exc}")
        return False


def _add_x_session_cookies(context) -> None:
    if not (X_AUTH_TOKEN and X_CT0):
        return
    context.add_cookies([
        {
            "name": "auth_token",
            "value": X_AUTH_TOKEN,
            "domain": ".x.com",
            "path": "/",
            "secure": True,
            "httpOnly": True,
            "sameSite": "None",
        },
        {
            "name": "ct0",
            "value": X_CT0,
            "domain": ".x.com",
            "path": "/",
            "secure": True,
            "httpOnly": False,
            "sameSite": "Lax",
        },
    ])


def _add_cookie_header_cookies(context, cookie_header: str, domain: str) -> None:
    if not cookie_header:
        return
    cookies = []
    for part in cookie_header.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        cookies.append({
            "name": name,
            "value": value,
            "domain": domain,
            "path": "/",
            "secure": True,
            "httpOnly": False,
            "sameSite": "Lax",
        })
    if cookies:
        context.add_cookies(cookies)


def _x_login_required(page) -> bool:
    try:
        if "/i/jf/onboarding/web" in page.url or "mode=login" in page.url:
            return True
        text = page.locator("body").inner_text(timeout=1000).lower()
        return "email or username" in text and "continue with" in text
    except Exception:
        return False


def _x_empty_reason(page) -> str:
    try:
        text = page.locator("body").inner_text(timeout=1000)
    except Exception:
        return "empty page"
    normalized = re.sub(r"\s+", " ", text).strip()
    lower = normalized.lower()
    if "something went wrong" in lower:
        return "x error page"
    if "try reloading" in lower:
        return "x reload prompt"
    if "no results" in lower:
        return "x no results page"
    if "email or username" in lower and "continue with" in lower:
        return "login page"
    return normalized[:180] or "empty page"


def _x_article_text(article) -> str:
    try:
        parts = article.locator("div[data-testid='tweetText']").all_inner_texts()
    except Exception:
        return ""
    text = " ".join(part.strip() for part in parts if part.strip())
    return re.sub(r"\s+", " ", text).strip()


def _metric_from_label(label: str) -> int:
    label = label.lower().replace(",", "")
    match = re.search(r"([\d.]+)\s*([km]?)", label)
    if not match:
        return 0
    value = float(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return int(value)


def _x_article_metrics(article) -> dict:
    metric_selectors = {
        "reply_count": "[data-testid='reply']",
        "retweet_count": "[data-testid='retweet']",
        "like_count": "[data-testid='like']",
    }
    metrics = {key: 0 for key in metric_selectors}
    for key, selector in metric_selectors.items():
        try:
            label = article.locator(selector).first.get_attribute("aria-label") or ""
            metrics[key] = _metric_from_label(label)
        except Exception:
            metrics[key] = 0
    return metrics


def _parse_rss(url: str) -> list[dict]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": REDDIT_USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        _mark_source("google_news", "ok")
    except Exception as exc:
        _mark_source("google_news", f"unavailable: {exc}")
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
    _reset_source_status()

    tickers = list(dict.fromkeys(tickers))
    stats = {ticker: MentionStats(ticker=ticker) for ticker in tickers}
    pattern = _ticker_pattern(tickers)

    # Determine if we need a browser session (Reddit blocked from this IP)
    reddit_posts = _fetch_reddit_posts()
    needs_browser = reddit_posts is None  # None means all non-browser strategies failed

    ranked_tickers = (
        scored_df.sort_values("prob_buy", ascending=False)["ticker"]
        .head(max(X_TOP_N, NEWS_TOP_N))
        .tolist()
    )

    if needs_browser:
        # All Reddit strategies failed — use shared browser for Reddit + X
        reddit_requests = _build_reddit_browser_requests()
        x_search_urls = [_x_search_url(t) for t in ranked_tickers[:X_TOP_N]]
        reddit_posts, x_posts = _fetch_all_browser_sources(reddit_requests, x_search_urls)
    else:
        # Reddit worked without browser — use X API independently
        x_posts = _fetch_x_posts(ranked_tickers)

    for post in (reddit_posts or []):
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

    for post in x_posts:
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

    out = pd.DataFrame([asdict(item) for item in stats.values()])
    out.attrs["source_status"] = dict(SOURCE_STATUS)
    return out


def apply_sentiment_overlay(scored_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored_df.empty:
        return scored_df, pd.DataFrame()
    overlay = build_sentiment_overlay(scored_df, scored_df["ticker"].tolist())
    source_status = overlay.attrs.get("source_status", dict(SOURCE_STATUS))
    if overlay.empty:
        scored_df = scored_df.copy()
        scored_df["base_prob_buy"] = scored_df["prob_buy"]
        scored_df["sentiment_adjustment"] = 0.0
        scored_df.attrs["source_status"] = source_status
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
    merged.attrs["source_status"] = source_status
    sorted_overlay = overlay.sort_values("adjusted_prob_buy", ascending=False).reset_index(drop=True)
    sorted_overlay.attrs["source_status"] = source_status
    return merged, sorted_overlay

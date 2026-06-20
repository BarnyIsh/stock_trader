"""
market_research.py - Dynamic universe screening and fundamental analysis

At each iteration this module:
  1. Screens S&P 500 + Nasdaq 100 tickers by volume/cap filters
  2. Ranks sectors using ETF momentum
  3. Scores each candidate on fundamental + technical merit
  4. Returns a ranked watchlist for the ML model to act on
"""

import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from config import (
    SECTOR_ETFS, MIN_AVG_VOLUME, MIN_MARKET_CAP_B,
    MIN_PRICE, MAX_PE_RATIO, MIN_PE_RATIO,
    MAX_PB_RATIO, MIN_ROE, MAX_DEBT_EQUITY
)


# ─── Universe loaders ────────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers — tries multiple sources, falls back to hardcoded list."""

    # Source 1: pull constituents directly from the SPY ETF holdings via yfinance
    try:
        spy = yf.Ticker("SPY")
        if hasattr(spy, "constituents") and spy.constituents is not None:
            tickers = list(spy.constituents.index)
            if len(tickers) > 400:
                print(f"  [universe] Loaded {len(tickers)} tickers from SPY constituents.")
                return [t.replace(".", "-") for t in tickers]
    except Exception:
        pass

    # Source 2: Wikipedia official REST API (no auth, no scraping, no 403)
    try:
        from io import StringIO
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":  "parse",
                "page":    "List of S&P 500 companies",
                "prop":    "text",
                "format":  "json",
                "section": "0",
            },
            headers={"User-Agent": "AlgoTrader/1.0 (educational project)"},
            timeout=15,
        )
        resp.raise_for_status()
        html = resp.json()["parse"]["text"]["*"]
        tables = pd.read_html(StringIO(html))
        # First table on the page is the constituent list
        for table in tables:
            for col in ["Symbol", "Ticker", "ticker", "symbol"]:
                if col in table.columns and len(table) > 400:
                    tickers = table[col].str.replace(".", "-", regex=False).tolist()
                    print(f"  [universe] Loaded {len(tickers)} tickers from Wikipedia API.")
                    return tickers
    except Exception as e:
        print(f"  [universe] Wikipedia API failed: {e}")

    # Source 3: hardcoded comprehensive list (top ~150 S&P 500 by market cap)
    print("  [universe] Using built-in ticker list.")
    return [
        # Mega-cap tech
        "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","AVGO","ORCL",
        # Financials
        "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","AXP","BLK","SPGI","CB",
        # Healthcare
        "UNH","LLY","JNJ","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN","ISRG",
        # Consumer
        "WMT","COST","PG","HD","MCD","KO","PEP","SBUX","NKE","TGT","LOW","TJX",
        # Industrials
        "CAT","BA","GE","HON","UPS","RTX","LMT","DE","MMM","ETN","ADP",
        # Energy
        "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY",
        # Communication
        "NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS","PARA",
        # Technology (non-mega)
        "AMD","INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","MRVL","ADI",
        "CRM","ADBE","NOW","INTU","PANW","SNPS","CDNS","FTNT","ANSS",
        # Other large-cap tech
        "ACN","IBM","CSCO","FICO","CTSH","IT","GLW",
        # Healthcare devices / biotech
        "SYK","BSX","MDT","EW","ZBH","BAX","DXCM","IDXX","A","IQV",
        # REITs / Utilities
        "NEE","DUK","SO","AEP","EXC","XEL","PCG","AMT","PLD","EQIX","CCI",
        # Materials
        "LIN","APD","SHW","ECL","NEM","FCX","NUE","VMC","MLM",
    ]


def get_nasdaq100_tickers() -> list[str]:
    """Fetch Nasdaq-100 tickers via multiple sources."""

    # Source 1: QQQ constituents via yfinance
    try:
        qqq = yf.Ticker("QQQ")
        if hasattr(qqq, "constituents") and qqq.constituents is not None:
            tickers = list(qqq.constituents.index)
            if len(tickers) > 80:
                return [t.replace(".", "-") for t in tickers]
    except Exception:
        pass

    # Source 2: Wikipedia official API for Nasdaq-100
    try:
        from io import StringIO
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":  "parse",
                "page":    "Nasdaq-100",
                "prop":    "text",
                "format":  "json",
                "section": "0",
            },
            headers={"User-Agent": "AlgoTrader/1.0 (educational project)"},
            timeout=15,
        )
        resp.raise_for_status()
        html = resp.json()["parse"]["text"]["*"]
        tables = pd.read_html(StringIO(html))
        for table in tables:
            for col in ["Ticker", "Symbol", "ticker", "symbol"]:
                if col in table.columns and len(table) > 80:
                    return table[col].str.replace(".", "-", regex=False).tolist()
    except Exception:
        pass

    return []


def get_full_universe() -> list[str]:
    sp = get_sp500_tickers()
    nq = get_nasdaq100_tickers()
    return list(dict.fromkeys(sp + nq))   # deduplicate, preserve order


# ─── Sector momentum ─────────────────────────────────────────────────────────

def rank_sectors(lookback_days: int = 20) -> pd.DataFrame:
    """
    Rank sectors by recent ETF momentum.
    Returns DataFrame with columns: sector, etf, momentum, rank.
    """
    rows = []
    etf_list = list(SECTOR_ETFS.values())
    data = yf.download(etf_list, period="3mo", interval="1d",
                       auto_adjust=True, progress=False)["Close"]

    for sector, etf in SECTOR_ETFS.items():
        if etf not in data.columns:
            continue
        prices = data[etf].dropna()
        if len(prices) < lookback_days:
            continue
        momentum = prices.iloc[-1] / prices.iloc[-lookback_days] - 1
        rows.append({"sector": sector, "etf": etf, "momentum": float(momentum)})

    df = pd.DataFrame(rows).sort_values("momentum", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


# ─── Fundamental scoring ─────────────────────────────────────────────────────

def score_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental data and return a score dict.
    Score components (each 0-1, higher = better):
      - value_score   : low P/E, low P/B relative to sector
      - quality_score : high ROE, low D/E
      - growth_score  : revenue + earnings growth
      - momentum_score: price vs 52-week range
    """
    try:
        info = yf.Ticker(ticker).info
        result = {
            "ticker":       ticker,
            "name":         info.get("longName", ticker),
            "sector":       info.get("sector", "Unknown"),
            "price":        info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "market_cap_b": (info.get("marketCap") or 0) / 1e9,
            "avg_volume":   info.get("averageVolume", 0),
            "pe_ratio":     info.get("trailingPE", None),
            "pb_ratio":     info.get("priceToBook", None),
            "roe":          info.get("returnOnEquity", None),
            "debt_equity":  info.get("debtToEquity", None),
            "rev_growth":   info.get("revenueGrowth", None),
            "earn_growth":  info.get("earningsGrowth", None),
            "52w_high":     info.get("fiftyTwoWeekHigh", None),
            "52w_low":      info.get("fiftyTwoWeekLow", None),
            "analyst_target": info.get("targetMeanPrice", None),
            "recommendation": info.get("recommendationKey", "none"),
        }

        # ── Liquidity gate ───────────────────────────────────────────────────
        if (
            result["price"] < MIN_PRICE
            or result["market_cap_b"] < MIN_MARKET_CAP_B
            or result["avg_volume"] < MIN_AVG_VOLUME
        ):
            result["passes_filter"] = False
            result["composite_score"] = 0.0
            return result

        result["passes_filter"] = True

        # ── Value score ──────────────────────────────────────────────────────
        v_scores = []
        pe = result["pe_ratio"]
        if pe and MIN_PE_RATIO < pe < MAX_PE_RATIO:
            v_scores.append(1 - pe / MAX_PE_RATIO)
        pb = result["pb_ratio"]
        if pb and 0 < pb < MAX_PB_RATIO:
            v_scores.append(1 - pb / MAX_PB_RATIO)
        result["value_score"] = float(np.mean(v_scores)) if v_scores else 0.3

        # ── Quality score ────────────────────────────────────────────────────
        q_scores = []
        roe = result["roe"]
        if roe and roe > 0:
            q_scores.append(min(roe / 0.30, 1.0))
        de = result["debt_equity"]
        if de is not None and de >= 0:
            q_scores.append(max(0, 1 - de / MAX_DEBT_EQUITY))
        result["quality_score"] = float(np.mean(q_scores)) if q_scores else 0.3

        # ── Growth score ─────────────────────────────────────────────────────
        g_scores = []
        rg = result["rev_growth"]
        if rg is not None:
            g_scores.append(min(max((rg + 0.1) / 0.3, 0), 1.0))
        eg = result["earn_growth"]
        if eg is not None:
            g_scores.append(min(max((eg + 0.1) / 0.4, 0), 1.0))
        result["growth_score"] = float(np.mean(g_scores)) if g_scores else 0.3

        # ── Price momentum vs 52-week range ──────────────────────────────────
        hi, lo = result["52w_high"], result["52w_low"]
        if hi and lo and hi != lo and result["price"]:
            pct_range = (result["price"] - lo) / (hi - lo)
            result["momentum_score"] = float(1 - abs(pct_range - 0.45) / 0.55)
        else:
            result["momentum_score"] = 0.3

        # ── Analyst signal bonus ─────────────────────────────────────────────
        rec_map = {
            "strongBuy": 0.2, "buy": 0.1, "hold": 0.0,
            "underperform": -0.1, "sell": -0.2,
        }
        analyst_bonus = rec_map.get(result["recommendation"], 0.0)

        # ── Upside to analyst target ──────────────────────────────────────────
        upside_score = 0.0
        if result["analyst_target"] and result["price"]:
            upside = (result["analyst_target"] - result["price"]) / result["price"]
            upside_score = min(max(upside / 0.30, 0), 0.2)

        # ── Composite ────────────────────────────────────────────────────────
        result["composite_score"] = (
            0.30 * result["value_score"]
            + 0.25 * result["quality_score"]
            + 0.25 * result["growth_score"]
            + 0.20 * result["momentum_score"]
            + analyst_bonus
            + upside_score
        )
        result["composite_score"] = float(
            np.clip(result["composite_score"], 0.0, 1.0)
        )
        return result

    except Exception as e:
        return {
            "ticker":          ticker,
            "passes_filter":   False,
            "composite_score": 0.0,
            "error":           str(e),
        }


# ─── Main research entry point ───────────────────────────────────────────────

def run_market_research(
    top_n_sectors: int = 5,
    max_candidates: int = 60,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full market research pipeline.
    Returns a ranked DataFrame of candidate stocks, best first.
    """
    if verbose:
        print("=" * 60)
        print("MARKET RESEARCH — " + datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("=" * 60)

    # 1. Sector momentum
    sectors_df = rank_sectors()
    top_sectors = set(sectors_df.head(top_n_sectors)["sector"].tolist())
    if verbose:
        print("\nTop sectors by momentum:")
        print(sectors_df.to_string(index=False))

    # 2. Universe
    universe = get_full_universe()
    if verbose:
        print(f"\nUniverse size: {len(universe)} tickers")

    # 3. Score candidates
    results = []
    checked = 0
    for ticker in universe:
        scored = score_fundamentals(ticker)
        if not scored.get("passes_filter", False):
            continue
        if scored.get("sector") in top_sectors:
            scored["composite_score"] = min(
                scored["composite_score"] + 0.05, 1.0
            )
            scored["sector_aligned"] = True
        else:
            scored["sector_aligned"] = False

        results.append(scored)
        checked += 1
        if verbose and checked % 20 == 0:
            print(f"  Scored {checked} candidates...")
        time.sleep(0.05)

        if len(results) >= max_candidates * 3:
            break

    if not results:
        print("[warn] No candidates passed filters.")
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("composite_score", ascending=False)
    df = df.head(max_candidates).reset_index(drop=True)

    if verbose:
        print(f"\nTop 15 candidates:")
        cols = ["ticker","name","sector","composite_score",
                "value_score","quality_score","growth_score"]
        available = [c for c in cols if c in df.columns]
        print(df[available].head(15).to_string(index=False))

    return df
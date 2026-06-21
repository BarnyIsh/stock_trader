"""
config.py - Central configuration for the algo trading system
"""
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─── Schwab API ──────────────────────────────────────────────────────────────
SCHWAB_APP_KEY      = os.getenv("SCHWAB_APP_KEY", "")
SCHWAB_APP_SECRET   = os.getenv("SCHWAB_APP_SECRET", "")
SCHWAB_CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1")

SCHWAB_AUTH_URL  = "https://api.schwabapi.com/v1/oauth/authorize"
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
SCHWAB_BASE_URL  = "https://api.schwabapi.com/trader/v1"
SCHWAB_MARKET_URL= "https://api.schwabapi.com/marketdata/v1"

# ─── Trading Mode ────────────────────────────────────────────────────────────
PAPER_TRADING       = os.getenv("PAPER_TRADING", "true").lower() == "true"
MAX_POSITION_SIZE   = float(os.getenv("MAX_POSITION_SIZE", "0.05"))
MAX_PORTFOLIO_STOCKS= int(os.getenv("MAX_PORTFOLIO_STOCKS", "20"))
RISK_PER_TRADE      = float(os.getenv("RISK_PER_TRADE", "0.02"))

# ─── Data Splits (strict train / test separation) ────────────────────────────
TRAINING_START = os.getenv("TRAINING_START", "2018-01-01")
TRAINING_END   = os.getenv("TRAINING_END",   "2022-12-31")
TEST_START     = os.getenv("TEST_START",     "2023-01-01")
TEST_END       = datetime.today().strftime("%Y-%m-%d")
BENCHMARK_TICKER = os.getenv("BENCHMARK_TICKER", "SPY")

# ─── Universe screener watchlists ────────────────────────────────────────────
# Sectors and sample tickers for initial universe — expanded dynamically
SECTOR_ETFS = {
    "Technology":    "XLK",
    "Healthcare":    "XLV",
    "Financials":    "XLF",
    "Energy":        "XLE",
    "Consumer Disc": "XLY",
    "Industrials":   "XLI",
    "Materials":     "XLB",
    "Utilities":     "XLU",
    "Real Estate":   "XLRE",
    "Comm Services": "XLC",
    "Staples":       "XLP",
}

# Minimum thresholds for a stock to enter the universe
MIN_AVG_VOLUME      = 500_000   # shares / day
MIN_MARKET_CAP_B    = 1.0       # $1 B
MIN_PRICE           = 5.0       # avoid penny stocks

# ─── Strategy parameters ─────────────────────────────────────────────────────
RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65
RSI_PERIOD     = 14

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

BB_PERIOD, BB_STD = 20, 2.0

# Fundamental thresholds (value-investing layer)
MAX_PE_RATIO    = 35
MIN_PE_RATIO    = 0
MAX_PB_RATIO    = 5
MIN_ROE         = 0.10   # 10%
MAX_DEBT_EQUITY = 2.0

# Model re-train cadence
RETRAIN_EVERY_N_DAYS = 30

# Score threshold above which we consider a stock a buy
BUY_SCORE_THRESHOLD  = 0.38
SELL_SCORE_THRESHOLD = 0.30

# Algorithmic Trading System — Schwab API

An end-to-end ML-powered trading bot that performs daily market research, trains on historical data, and trades exclusively using unseen data via the Schwab Developer API.

---

## Architecture

```
trader.py (orchestrator)
├── market_research.py   → screens universe, ranks sectors, scores fundamentals
├── features.py          → technical indicator engineering (no future leakage)
├── model.py             → train on 2018-2022, test on 2023-present (unseen)
├── portfolio.py         → Kelly sizing, ATR stops, circuit breaker, rebalancing
└── schwab_client.py     → OAuth2 auth + Schwab API (quotes, orders, account)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your Schwab credentials
```

### 3. Register a Schwab Developer App
1. Go to [developer.schwab.com](https://developer.schwab.com)
2. Create an app → get **App Key** and **App Secret**
3. Set Callback URL to `https://127.0.0.1`
4. Add credentials to `.env`

### 4. Authenticate (OAuth2)
```bash
python trader.py --mode auth
```
Follow the browser flow. Tokens are saved to `.schwab_tokens.json` and auto-refresh.

---

## Usage

### Train the model (historical data: 2018–2022)
```bash
python trader.py --mode train
```
- Screens S&P 500 + Nasdaq 100 universe
- Runs walk-forward cross-validation (no data leakage)
- Trains RF + GradientBoosting + LogisticRegression ensemble
- Saves models to `models/`

### Backtest on unseen data (2023–present)
```bash
python trader.py --mode backtest
```
- Loads trained models
- Evaluates **exclusively** on post-2022 data the model has never seen
- Reports ROC-AUC, precision, recall, simulated P&L

### Run daily trading session
```bash
python trader.py --mode run
```
- Runs market research to find best sectors/stocks
- Scores candidates with ensemble
- Fetches live Schwab prices
- Buys undervalued/high-score stocks
- Sells overvalued/degraded-score positions
- Executes orders via Schwab API

### Schedule for 9:31 AM ET daily
```bash
python scheduler.py          # foreground daemon
# OR add to crontab:
python scheduler.py --cron   # prints the crontab line
```

---

## Strategy Logic

### Market Research (daily)
1. Fetch S&P 500 + Nasdaq 100 tickers
2. Rank sectors by 20-day ETF momentum
3. Score each stock on:
   - **Value**: P/E, P/B ratios vs thresholds
   - **Quality**: ROE, Debt/Equity
   - **Growth**: Revenue + earnings growth rates
   - **Momentum**: price vs 52-week range
   - **Analyst consensus**: analyst recommendations + price targets
4. Sector-aligned stocks get a +5% bonus

### ML Model (trained once, tested on unseen)
- **Features (31 total)**: RSI, MACD, Bollinger Bands, ATR, OBV, MFI, ADX, Stochastic, EMA crosses, price vs 52-week extremes, candlestick patterns
- **Label**: 1 if 5-day forward return ≥ 1.5%
- **Ensemble**: RF (40%) + GradientBoosting (40%) + LogisticRegression (20%)
- **Validation**: `TimeSeriesSplit` walk-forward CV within training window only
- **Train cutoff**: Dec 31, 2022 — model never sees 2023+ during training

### Position Sizing
- **Half-Kelly criterion** based on model probability
- Capped at 5% of portfolio per position
- **ATR-based stop loss**: entry − 2 × ATR
- **Price target**: entry + 3 × ATR (3:1 reward-risk)
- Max 20 simultaneous positions

### Risk Management
- **Circuit breaker**: if portfolio drawdown ≥ 20%, sell everything and halt
- **Stop-loss**: auto-sell if price hits ATR-derived stop
- **Target-taking**: auto-sell when price target reached
- **Score degradation**: sell if combined ML + fundamental score drops below 0.40
- **Concentration limit**: max 5% per position

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `SCHWAB_APP_KEY` | — | From Schwab Developer portal |
| `SCHWAB_APP_SECRET` | — | From Schwab Developer portal |
| `PAPER_TRADING` | `true` | Set `false` for real orders |
| `MAX_POSITION_SIZE` | `0.05` | Max 5% of portfolio per stock |
| `MAX_PORTFOLIO_STOCKS` | `20` | Max simultaneous positions |
| `RISK_PER_TRADE` | `0.02` | Risk 2% of capital per trade |
| `TRAINING_START` | `2018-01-01` | Model training start |
| `TRAINING_END` | `2022-12-31` | Model training end (strict) |
| `TEST_START` | `2023-01-01` | Unseen test data begins here |

---

## Files

```
algo_trader/
├── trader.py            # Main orchestrator — run this
├── scheduler.py         # Market-hours scheduler
├── schwab_client.py     # Schwab API client (OAuth2 + orders)
├── market_research.py   # Stock screener + fundamental analysis
├── features.py          # Technical feature engineering
├── model.py             # ML ensemble training + inference
├── portfolio.py         # Position sizing + risk management
├── config.py            # All constants and thresholds
├── .env.example         # Config template
├── models/              # Saved model files + metadata
└── logs/                # Daily session logs
```

---

## ⚠️ Important Warnings

1. **Paper trade first.** Set `PAPER_TRADING=true` until you've backtested and are confident in performance.
2. **Past performance ≠ future results.** ML models can decay; retrain regularly (default: every 30 days).
3. **This is not financial advice.** You are responsible for all trading decisions.
4. **Market hours**: The bot is designed to run at 9:31 AM ET on weekdays. Running outside market hours will queue orders for next open.
5. **API rate limits**: The Schwab API has rate limits. The system throttles requests automatically.

---

## Schwab API Reference
- Docs: https://developer.schwab.com/products/trader-api--individual-
- Auth: OAuth2 Authorization Code flow
- Base URL: `https://api.schwabapi.com/trader/v1`
- Market Data: `https://api.schwabapi.com/marketdata/v1`

"""
trader.py - Main orchestrator for the algo trading system

Daily workflow:
  1. Run market research → find best sectors / stocks
  2. Score today's candidates with ML model
  3. Fetch live prices + ATR from Schwab API
  4. Portfolio manager decides buys/sells
  5. Execute orders via Schwab API
  6. Save state + log results

Run modes:
  python trader.py --mode train       # train model on historical data
  python trader.py --mode backtest    # evaluate on unseen test data
  python trader.py --mode run         # live daily session
  python trader.py --mode auth        # OAuth2 setup wizard
"""

import argparse
import json
import os
import time
import pandas as pd
from datetime    import datetime
from pathlib     import Path

from config          import PAPER_TRADING, RETRAIN_EVERY_N_DAYS, TEST_START
from market_research import run_market_research
from features        import download_ohlcv, add_technical_features, FEATURE_COLS
from model           import train_model, evaluate_on_test, score_today
from portfolio       import Portfolio
from schwab_client   import SchwabClient


LOG_DIR = (
    Path("/tmp/stock_trader_logs")
    if os.getenv("VERCEL")
    else Path(__file__).parent / "logs"
)
LOG_DIR.mkdir(exist_ok=True)
LOG_INTENT_HOLD_DAYS = 5
LOG_REBUY_WAIT_DAYS = 10


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_atr_map(tickers: list[str], client: SchwabClient | None = None) -> dict:
    """Fetch ATR for each ticker from recent price history."""
    atr_map = {}
    for ticker in tickers:
        try:
            df = download_ohlcv(ticker, start=TEST_START, end=datetime.today().strftime("%Y-%m-%d"))
            if len(df) < 20:
                continue
            df = add_technical_features(df)
            if "atr" in df.columns and not df["atr"].isna().all():
                atr_map[ticker] = float(df["atr"].iloc[-1])
        except Exception:
            pass
    return atr_map


def get_live_prices(
    tickers: list[str],
    client: SchwabClient | None = None,
) -> dict[str, float]:
    """
    Fetch real-time prices.
    If Schwab client is available and authenticated, use the API.
    Otherwise fall back to yfinance last-close.
    """
    prices = {}
    if client and client.tokens:
        try:
            quotes = client.get_quotes(tickers)
            for sym, data in quotes.items():
                q = data.get("quote", {})
                prices[sym] = q.get("lastPrice") or q.get("closePrice", 0)
            return prices
        except Exception as e:
            print(f"[warn] Schwab quote fetch failed: {e}. Falling back to yfinance.")

    # yfinance fallback
    import yfinance as yf
    data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    for col in data.columns:
        val = data[col].dropna().iloc[-1] if not data[col].dropna().empty else 0
        prices[col] = float(val)
    return prices


def append_trade_execution_log(entries: list[dict]):
    """Append paper/live trade attempts to a durable JSONL audit log."""
    if not entries:
        return
    path = LOG_DIR / "trades.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")


# ─── OAuth2 setup ─────────────────────────────────────────────────────────────

def run_auth_wizard(client: SchwabClient):
    print("\n" + "="*60)
    print("SCHWAB OAUTH2 SETUP")
    print("="*60)
    print("\n1. Open this URL in your browser and log in:\n")
    print(client.get_auth_url())
    print("\n2. After authorizing, Schwab will redirect you to:")
    print("   https://127.0.0.1?code=XXXXXX&session=YYYYYY")
    print("\n3. Paste the FULL redirect URL here:")
    redirect = input("URL: ").strip()

    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(redirect)
    code   = parse_qs(parsed.query).get("code", [""])[0]
    if not code:
        print("[error] Could not extract code from URL.")
        return

    tokens = client.exchange_code(code)
    print(f"\n✅ Authenticated! Tokens saved. Access token expires in "
          f"{tokens.get('expires_in', '?')} seconds.")
    accs = client.get_accounts()
    print(f"   Linked accounts: {len(accs)}")
    for a in accs:
        val = a["securitiesAccount"]["currentBalances"].get("liquidationValue", 0)
        print(f"   • {a['securitiesAccount']['accountNumber']}  ${val:,.2f}")


# ─── Train mode ───────────────────────────────────────────────────────────────

def run_training():
    print("\n📊 Fetching stock universe for training…")
    research_df = run_market_research(max_candidates=200, verbose=False)
    if research_df.empty:
        print("[error] No candidates found.")
        return
    tickers = research_df["ticker"].tolist()[:150]   # large universe for better generalization
    print(f"Training on {len(tickers)} stocks: {tickers[:10]}…")
    bundle = train_model(tickers, forward_days=5, min_return=0.015, verbose=True)
    print("\n✅ Training complete.")
    return bundle


# ─── Backtest mode ────────────────────────────────────────────────────────────

def run_backtest(bundle=None):
    if bundle is None:
        # Load from disk
        import joblib, json
        from pathlib import Path
        model_dir = Path(__file__).parent / "models"
        rf   = joblib.load(model_dir / "rf_latest.pkl")
        gb   = joblib.load(model_dir / "gb_latest.pkl")
        lr   = joblib.load(model_dir / "lr_latest.pkl")
        meta = json.loads((model_dir / "metadata.json").read_text())
        bundle = {"models": {"rf": rf, "gb": gb, "lr": lr}, "metadata": meta}

    tickers = bundle["metadata"]["tickers"][:40]
    test_df = evaluate_on_test(bundle, tickers, verbose=True)

    if not test_df.empty:
        # Simple backtest P&L simulation
        simulate_backtest(test_df)

    return test_df


def simulate_backtest(test_df: pd.DataFrame, forward_days: int = 5):
    """
    Simulate P&L from model signals on test data.
    Buys when signal=1, holds forward_days, then sells.
    """
    print("\n" + "="*60)
    print("BACKTEST SIMULATION (equal-weight, no transaction costs)")
    print("="*60)

    signals = test_df[test_df["signal"] == 1].copy()
    if signals.empty:
        print("No buy signals in test period.")
        return

    # Use actual forward_return as the realised P&L
    if "future_return" not in signals.columns:
        print("No future_return column. Skipping simulation.")
        return

    returns = signals["future_return"].dropna()
    print(f"Total signals:  {len(signals)}")
    print(f"Win rate:       {(returns > 0).mean():.1%}")
    print(f"Avg return:     {returns.mean():.2%}")
    print(f"Median return:  {returns.median():.2%}")
    print(f"Best trade:     {returns.max():.2%}")
    print(f"Worst trade:    {returns.min():.2%}")
    print(f"Sharpe (raw):   {returns.mean() / returns.std():.2f}")

    # Monthly breakdown
    signals["date"] = pd.to_datetime(signals["date"])
    signals["month"] = signals["date"].dt.to_period("M")
    monthly = signals.groupby("month")["future_return"].mean()
    print("\nMonthly avg return (signalled trades):")
    print(monthly.to_string())


# ─── Live run mode ────────────────────────────────────────────────────────────

def run_daily_session(client: SchwabClient):
    print("\n" + "="*60)
    print(f"DAILY SESSION — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if PAPER_TRADING:
        print("⚠️  PAPER TRADING MODE (no real orders)")
    print("="*60)

    # 1. Market research
    print("\n[1/5] Running market research…")
    research_df = run_market_research(max_candidates=60, verbose=True)
    if research_df.empty:
        print("[error] Market research returned no results.")
        return
    tickers = research_df["ticker"].tolist()

    # 2. Check if model needs retraining
    meta_path = Path(__file__).parent / "models" / "metadata.json"
    needs_train = not meta_path.exists()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        trained_at = datetime.strptime(meta["trained_at"], "%Y%m%d_%H%M")
        days_since  = (datetime.now() - trained_at).days
        if days_since >= RETRAIN_EVERY_N_DAYS:
            print(f"  Model is {days_since} days old → retraining…")
            needs_train = True

    bundle = None
    if needs_train:
        bundle = run_training()

    # 3. Score today's candidates
    print("\n[2/5] Scoring candidates with ML model…")
    try:
        scored_df = score_today(tickers, model_bundle=bundle)
        print(f"  Scored {len(scored_df)} tickers. Top 5:")
        if not scored_df.empty:
            print(scored_df[["ticker","prob_buy","rsi","bb_pct"]].head(5).to_string(index=False))
    except RuntimeError as e:
        print(f"  {e} — triggering training first.")
        bundle = run_training()
        scored_df = score_today(tickers, model_bundle=bundle)

    # 4. Fetch live prices and ATR
    print("\n[3/5] Fetching live prices…")
    all_tickers = list(set(
        tickers
        + list(Portfolio.load().positions.keys())
    ))
    prices = get_live_prices(all_tickers, client)
    atr_map = get_atr_map(tickers)

    # 5. Portfolio decisions
    print("\n[4/5] Computing portfolio decisions…")
    portfolio = Portfolio.load()
    decisions = portfolio.decide_trades(
        scored_df=scored_df,
        research_df=research_df,
        current_prices=prices,
        atr_map=atr_map,
    )
    score_map = (
        dict(zip(scored_df["ticker"], scored_df["prob_buy"]))
        if not scored_df.empty else {}
    )

    # 6. Execute
    print(f"\n[5/5] Executing trades "
          f"({'PAPER' if PAPER_TRADING else 'LIVE'})…")

    # Write human-readable intent log before executing trades
    intent_lines = []
    intent_lines.append("TRADE INTENT SUMMARY")
    intent_lines.append(f"date: {datetime.now().isoformat()}")
    intent_lines.append(f"portfolio_value: {decisions['portfolio_value']:.2f}")
    intent_lines.append(f"drawdown: {decisions['drawdown']:.2%}")
    intent_lines.append("")
    intent_lines.append("SELL INTENTS:")
    if decisions["sells"]:
        for ticker, shares, reason in decisions["sells"]:
            intent_lines.append(
                f"  SELL {shares} × {ticker} now — reason: {reason} — "
                f"planned BUY back after {LOG_REBUY_WAIT_DAYS} days or when model signals BUY"
            )
    else:
        intent_lines.append("  (none)")

    intent_lines.append("")
    intent_lines.append("BUY INTENTS:")
    if decisions["buys"]:
        for ticker, shares, limit, stop, target in decisions["buys"]:
            intent_lines.append(
                f"  BUY  {shares} × {ticker} @ ${limit:.2f}  stop: ${stop:.2f}  "
                f"target: ${target:.2f}  — planned SELL in {LOG_INTENT_HOLD_DAYS} days or when target reached"
            )
    else:
        intent_lines.append("  (none)")

    intent_lines.append("")
    intent_lines.append("Notes: 'planned' timings are heuristics for human review only.")

    intent_file = LOG_DIR / f"intent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    intent_file.write_text("\n".join(intent_lines))
    print(f"\nIntent log saved → {intent_file}")

    print(f"\n  → Sells ({len(decisions['sells'])}):")
    if decisions["sells"]:
        portfolio.apply_sells(decisions["sells"], prices)
        if not PAPER_TRADING:
            for ticker, shares, reason in decisions["sells"]:
                try:
                    client.place_market_order(ticker, shares, action="SELL")
                except Exception as e:
                    print(f"    [err] SELL {ticker}: {e}")
    else:
        print("  (none)")

    print(f"\n  → Buys ({len(decisions['buys'])}):")
    if decisions["buys"]:
        portfolio.apply_buys(decisions["buys"], prices, score_map)
        if not PAPER_TRADING:
            for ticker, shares, limit, stop, target in decisions["buys"]:
                try:
                    client.place_limit_order(ticker, shares, limit, action="BUY")
                except Exception as e:
                    print(f"    [err] BUY {ticker}: {e}")
    else:
        print("  (none)")

    execution_log = []
    for ticker, shares, reason in decisions["sells"]:
        execution_log.append({
            "date": datetime.now().isoformat(),
            "mode": "PAPER" if PAPER_TRADING else "LIVE",
            "action": "SELL",
            "ticker": ticker,
            "shares": int(shares),
            "price": float(prices.get(ticker, 0)),
            "order_type": "MARKET",
            "reason": reason,
            "status": "paper" if PAPER_TRADING else "submitted_or_attempted",
        })
    for ticker, shares, limit, stop, target in decisions["buys"]:
        execution_log.append({
            "date": datetime.now().isoformat(),
            "mode": "PAPER" if PAPER_TRADING else "LIVE",
            "action": "BUY",
            "ticker": ticker,
            "shares": int(shares),
            "price": float(prices.get(ticker, limit)),
            "order_type": "LIMIT",
            "limit_price": float(limit),
            "stop": float(stop),
            "target": float(target),
            "status": "paper" if PAPER_TRADING else "submitted_or_attempted",
        })
    append_trade_execution_log(execution_log)

    portfolio.save()
    portfolio.print_summary(prices)

    # Log session
    log_entry = {
        "date":           datetime.now().isoformat(),
        "portfolio_value":decisions["portfolio_value"],
        "drawdown":       decisions["drawdown"],
        "buys":           [(b[0], b[1], b[2]) for b in decisions["buys"]],
        "sells":          [(s[0], s[1], s[2]) for s in decisions["sells"]],
        "executions":     execution_log,
    }
    log_file = LOG_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.write_text(json.dumps(log_entry, indent=2))
    print(f"\nSession log saved → {log_file}")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Algo Trading Bot (Schwab)")
    parser.add_argument(
        "--mode",
        choices=["auth", "train", "backtest", "run"],
        default="run",
        help="Operating mode",
    )
    args = parser.parse_args()

    client = SchwabClient()

    if args.mode == "auth":
        run_auth_wizard(client)

    elif args.mode == "train":
        run_training()

    elif args.mode == "backtest":
        run_backtest()

    elif args.mode == "run":
        run_daily_session(client)


if __name__ == "__main__":
    main()

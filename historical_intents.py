"""
historical_intents.py - Reconstruct daily model trade intents at market open.

For each requested trading day, this script uses features available as of the
previous close, prices trades at that day's open, and records the buy/sell
orders the portfolio logic would have wanted to make.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

from config import BENCHMARK_TICKER
from features import build_feature_matrix, FEATURE_COLS
from model import MODEL_DIR
from portfolio import Portfolio


LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRADE_COLUMNS = [
    "trade_date", "signal_date", "action", "ticker", "shares", "price",
    "limit_price", "stop", "target", "reason", "prob_buy",
]


def load_model_bundle() -> dict:
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return {
        "models": {
            "rf": joblib.load(MODEL_DIR / "rf_latest.pkl"),
            "gb": joblib.load(MODEL_DIR / "gb_latest.pkl"),
            "lr": joblib.load(MODEL_DIR / "lr_latest.pkl"),
        },
        "metadata": meta,
    }


def ensemble_prob(models: dict, X: pd.DataFrame) -> pd.Series:
    p_rf = models["rf"].predict_proba(X)[:, 1]
    p_gb = models["gb"].predict_proba(X)[:, 1]
    p_lr = models["lr"].predict_proba(X)[:, 1]
    return pd.Series(0.40 * p_rf + 0.40 * p_gb + 0.20 * p_lr, index=X.index)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    clean = df.copy()
    clean = clean.fillna("")
    headers = list(clean.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in clean.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def get_trading_days(days: int, end_date: str | None = None) -> list[pd.Timestamp]:
    end = pd.Timestamp(end_date or datetime.today().date()) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=max(220, days * 3))
    spy = yf.download(
        BENCHMARK_TICKER,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if spy.empty:
        raise RuntimeError(f"No {BENCHMARK_TICKER} trading calendar data found.")
    return list(pd.to_datetime(spy.index).tz_localize(None)[-days:])


def download_open_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No historical open prices downloaded.")
    if isinstance(data.columns, pd.MultiIndex):
        opens = data["Open"].copy()
    else:
        opens = data[["Open"]].rename(columns={"Open": tickers[0]})
    opens.index = pd.to_datetime(opens.index).tz_localize(None)
    return opens


def run_historical_intents(days: int = 100, end_date: str | None = None) -> dict:
    bundle = load_model_bundle()
    tickers = bundle["metadata"]["tickers"]
    feature_cols = bundle["metadata"]["feature_cols"]
    models = bundle["models"]

    trading_days = get_trading_days(days, end_date=end_date)
    first_day = trading_days[0]
    last_day = trading_days[-1]
    feature_start = (first_day - pd.Timedelta(days=520)).strftime("%Y-%m-%d")
    data_end = (last_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    features_df = build_feature_matrix(
        tickers,
        start=feature_start,
        end=data_end,
        label=False,
    )
    if features_df.empty:
        raise RuntimeError("No feature rows built for historical intent run.")

    features_df["date"] = pd.to_datetime(features_df["date"]).dt.tz_localize(None)
    X = features_df[feature_cols].fillna(0)
    features_df["prob_buy"] = ensemble_prob(models, X)

    open_prices = download_open_prices(
        tickers,
        start=first_day.strftime("%Y-%m-%d"),
        end=(last_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    portfolio = Portfolio(cash=100_000.0, peak_value=100_000.0)
    daily_rows = []
    trade_rows = []

    for trade_day in trading_days:
        available = features_df[features_df["date"] < trade_day]
        if available.empty:
            continue
        signal_date = available["date"].max()
        scored_df = available[available["date"] == signal_date].copy()
        scored_df["signal_date"] = signal_date
        scored_df["price"] = scored_df["close"]
        scored_df["signal"] = (scored_df["prob_buy"] >= 0.55).astype(int)

        if trade_day not in open_prices.index:
            continue
        prices = {
            ticker: float(price)
            for ticker, price in open_prices.loc[trade_day].dropna().items()
            if float(price) > 0
        }
        atr_map = dict(zip(scored_df["ticker"], scored_df.get("atr", pd.Series())))

        decisions = portfolio.decide_trades(
            scored_df=scored_df,
            research_df=pd.DataFrame(),
            current_prices=prices,
            atr_map=atr_map,
        )

        for ticker, shares, reason in decisions["sells"]:
            price = float(prices.get(ticker, 0))
            trade_rows.append({
                "trade_date": trade_day.date().isoformat(),
                "signal_date": signal_date.date().isoformat(),
                "action": "SELL",
                "ticker": ticker,
                "shares": int(shares),
                "price": price,
                "reason": reason,
                "prob_buy": float(scored_df.loc[
                    scored_df["ticker"] == ticker, "prob_buy"
                ].iloc[0]) if ticker in set(scored_df["ticker"]) else None,
            })

        for ticker, shares, limit, stop, target in decisions["buys"]:
            row = scored_df[scored_df["ticker"] == ticker].iloc[0]
            price = float(prices.get(ticker, limit))
            trade_rows.append({
                "trade_date": trade_day.date().isoformat(),
                "signal_date": signal_date.date().isoformat(),
                "action": "BUY",
                "ticker": ticker,
                "shares": int(shares),
                "price": price,
                "limit_price": float(limit),
                "stop": float(stop),
                "target": float(target),
                "prob_buy": float(row["prob_buy"]),
            })

        portfolio.apply_sells(decisions["sells"], prices)
        score_map = dict(zip(scored_df["ticker"], scored_df["prob_buy"]))
        portfolio.apply_buys(decisions["buys"], prices, score_map)

        daily_rows.append({
            "trade_date": trade_day.date().isoformat(),
            "signal_date": signal_date.date().isoformat(),
            "portfolio_value_at_open": decisions["portfolio_value"],
            "cash_before_trades": decisions["cash"],
            "drawdown": decisions["drawdown"],
            "buy_count": len(decisions["buys"]),
            "sell_count": len(decisions["sells"]),
            "top_signal_ticker": scored_df.sort_values(
                "prob_buy", ascending=False
            ).iloc[0]["ticker"],
            "top_signal_prob_buy": float(scored_df["prob_buy"].max()),
            "positions_after_trades": len(portfolio.positions),
            "cash_after_trades": portfolio.cash,
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    daily_df = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)

    daily_path = LOG_DIR / f"historical_intents_daily_{ts}.csv"
    trades_path = LOG_DIR / f"historical_intents_trades_{ts}.csv"
    json_path = LOG_DIR / f"historical_intents_{ts}.json"
    md_path = LOG_DIR / f"historical_intents_summary_{ts}.md"

    daily_df.to_csv(daily_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    summary = {
        "generated_at": ts,
        "days_requested": days,
        "days_documented": int(len(daily_df)),
        "first_trade_date": daily_df["trade_date"].min() if not daily_df.empty else None,
        "last_trade_date": daily_df["trade_date"].max() if not daily_df.empty else None,
        "model_trained_at": bundle["metadata"].get("trained_at"),
        "tickers": tickers,
        "buy_count": int((trades_df["action"] == "BUY").sum()) if not trades_df.empty else 0,
        "sell_count": int((trades_df["action"] == "SELL").sum()) if not trades_df.empty else 0,
        "final_cash": float(portfolio.cash),
        "final_positions": {
            ticker: pos.to_dict()
            for ticker, pos in portfolio.positions.items()
        },
        "daily_csv": str(daily_path),
        "trades_csv": str(trades_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        "# Historical Market-Open Trade Intents",
        "",
        f"Generated: {ts}",
        f"Model trained at: {summary['model_trained_at']}",
        f"Trading days documented: {summary['days_documented']}",
        f"Date range: {summary['first_trade_date']} to {summary['last_trade_date']}",
        f"Buys: {summary['buy_count']}",
        f"Sells: {summary['sell_count']}",
        f"Final simulated cash: ${summary['final_cash']:,.2f}",
        f"Open positions after run: {len(portfolio.positions)}",
        "",
        "This uses prior-close model signals and same-day open prices.",
        "The live buy threshold remains the configured portfolio threshold.",
        "",
        "## Files",
        f"- Daily summary CSV: `{daily_path.name}`",
        f"- Trade intents CSV: `{trades_path.name}`",
        f"- JSON summary: `{json_path.name}`",
        "",
        "## Recent Trade Intents",
    ]
    if trades_df.empty:
        lines.append("No buy or sell intents were generated.")
    else:
        recent = trades_df.tail(25).copy()
        lines.append(markdown_table(recent))
    lines.extend(["", "## Strongest Daily Signals"])
    if daily_df.empty:
        lines.append("No daily signal rows were generated.")
    else:
        strongest = daily_df.sort_values(
            "top_signal_prob_buy", ascending=False
        ).head(15)
        lines.append(markdown_table(strongest[[
            "trade_date", "signal_date", "top_signal_ticker",
            "top_signal_prob_buy", "buy_count", "sell_count",
        ]]))
    md_path.write_text("\n".join(lines))
    summary["summary_md"] = str(md_path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()
    summary = run_historical_intents(days=args.days, end_date=args.end_date)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

"""
market_open_job.py - Serverless-friendly daily market-open intent email.

This module avoids Schwab order execution. It only scores the latest trained
model, computes paper buy/sell intents, and emails the result.
"""

import json
import os
import smtplib
from dataclasses import asdict
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from config import PAPER_TRADING
from model import MODEL_DIR, score_today
from portfolio import Portfolio, Position
from trader import get_atr_map, get_live_prices


LOG_DIR = (
    Path("/tmp/stock_trader_logs")
    if os.getenv("VERCEL")
    else Path(__file__).parent / "logs"
)
LOG_DIR.mkdir(exist_ok=True)
DEFAULT_EMAIL_TO = "bryan.g.shi@gmail.com"


def _load_model_bundle() -> dict:
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return {
        "models": {
            "rf": joblib.load(MODEL_DIR / "rf_latest.pkl"),
            "gb": joblib.load(MODEL_DIR / "gb_latest.pkl"),
            "lr": joblib.load(MODEL_DIR / "lr_latest.pkl"),
        },
        "metadata": meta,
    }


def _portfolio_from_env() -> Portfolio:
    raw = os.getenv("PORTFOLIO_STATE_JSON", "").strip()
    if not raw:
        return Portfolio.load()

    state = json.loads(raw)
    portfolio = Portfolio(
        cash=float(state.get("cash", 100_000.0)),
        peak_value=float(state.get("peak_value", state.get("cash", 100_000.0))),
    )
    portfolio.positions = {
        ticker: Position(**position)
        for ticker, position in state.get("positions", {}).items()
    }
    portfolio.trade_log = state.get("trade_log", [])
    return portfolio


def _plain_trade_table(rows: list[dict]) -> str:
    if not rows:
        return "  (none)"
    lines = []
    for row in rows:
        if row["action"] == "BUY":
            lines.append(
                f"  BUY  {row['shares']:>5} {row['ticker']:<6} "
                f"@ ${row['price']:.2f}  prob={row.get('prob_buy', 0):.3f}  "
                f"stop=${row.get('stop', 0):.2f} target=${row.get('target', 0):.2f}"
            )
        else:
            lines.append(
                f"  SELL {row['shares']:>5} {row['ticker']:<6} "
                f"@ ${row['price']:.2f}  reason={row.get('reason', '')}"
            )
    return "\n".join(lines)


def _send_email(subject: str, body: str):
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    email_to = os.getenv("EMAIL_TO", DEFAULT_EMAIL_TO).strip()
    email_from = os.getenv("EMAIL_FROM", smtp_user or email_to).strip()

    if not smtp_host or not smtp_user or not smtp_password:
        raise RuntimeError(
            "Missing SMTP_HOST, SMTP_USER, or SMTP_PASSWORD environment variables."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)


def run_market_open_job(send_email: bool = True) -> dict:
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    bundle = _load_model_bundle()
    tickers = bundle["metadata"]["tickers"]

    scored_df = score_today(tickers, model_bundle=bundle)
    if scored_df.empty:
        raise RuntimeError("Model scoring returned no rows.")

    portfolio = _portfolio_from_env()
    current_tickers = list(dict.fromkeys(tickers + list(portfolio.positions.keys())))
    prices = get_live_prices(current_tickers, client=None)
    atr_map = get_atr_map(tickers)

    decisions = portfolio.decide_trades(
        scored_df=scored_df,
        research_df=pd.DataFrame(),
        current_prices=prices,
        atr_map=atr_map,
    )

    score_map = dict(zip(scored_df["ticker"], scored_df["prob_buy"]))
    trade_rows = []
    for ticker, shares, reason in decisions["sells"]:
        trade_rows.append({
            "action": "SELL",
            "ticker": ticker,
            "shares": int(shares),
            "price": float(prices.get(ticker, 0)),
            "reason": reason,
            "prob_buy": float(score_map.get(ticker, 0.0)),
        })
    for ticker, shares, limit, stop, target in decisions["buys"]:
        trade_rows.append({
            "action": "BUY",
            "ticker": ticker,
            "shares": int(shares),
            "price": float(prices.get(ticker, limit)),
            "limit_price": float(limit),
            "stop": float(stop),
            "target": float(target),
            "prob_buy": float(score_map.get(ticker, 0.0)),
        })

    top = scored_df.head(10)[["ticker", "price", "prob_buy", "rsi", "bb_pct"]]
    subject = f"Stock Trader intents for {now_ny:%Y-%m-%d}"
    body = "\n".join([
        f"Market-open model intents for {now_ny:%Y-%m-%d %H:%M %Z}",
        "",
        f"Mode: {'PAPER' if PAPER_TRADING else 'LIVE-INTENT-ONLY'}",
        f"Portfolio value used for sizing: ${decisions['portfolio_value']:,.2f}",
        f"Cash used for sizing: ${decisions['cash']:,.2f}",
        f"Drawdown: {decisions['drawdown']:.2%}",
        "",
        "Trades the model wants:",
        _plain_trade_table(trade_rows),
        "",
        "Top model scores:",
        top.to_string(index=False),
        "",
        "This email is an intent log only; it does not place orders.",
    ])

    log_entry = {
        "date": now_ny.isoformat(),
        "trade_rows": trade_rows,
        "decisions": {
            "portfolio_value": decisions["portfolio_value"],
            "cash": decisions["cash"],
            "drawdown": decisions["drawdown"],
        },
        "top_scores": top.to_dict(orient="records"),
        "positions": {
            ticker: asdict(position)
            for ticker, position in portfolio.positions.items()
        },
    }
    log_file = LOG_DIR / f"market_open_intents_{now_ny:%Y%m%d_%H%M%S}.json"
    log_file.write_text(json.dumps(log_entry, indent=2, default=str))

    if send_email:
        _send_email(subject, body)

    return {
        "ok": True,
        "date": now_ny.isoformat(),
        "email_sent": send_email,
        "trade_count": len(trade_rows),
        "buy_count": sum(1 for row in trade_rows if row["action"] == "BUY"),
        "sell_count": sum(1 for row in trade_rows if row["action"] == "SELL"),
        "log_file": str(log_file),
        "trades": trade_rows,
    }

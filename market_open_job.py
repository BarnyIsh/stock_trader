"""
market_open_job.py - Serverless-friendly daily market-open intent email.

This module avoids Schwab order execution. It only scores the latest trained
model, computes paper buy/sell intents, and emails the result.
"""

import json
import os
import smtplib
from html import escape
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
from sentiment_overlay import apply_sentiment_overlay
from trader import get_atr_map, get_live_prices


LOG_DIR = (
    Path("/tmp/stock_trader_logs")
    if os.getenv("VERCEL")
    else Path(__file__).parent / "logs"
)
LOG_DIR.mkdir(exist_ok=True)

# Subscription-based email list.
# EMAIL_SUBSCRIBERS is a comma-separated list of email addresses.
# Falls back to EMAIL_TO for backward compat, then to the default.
DEFAULT_SUBSCRIBERS = "bryan.g.shi@gmail.com"


def _get_subscribers() -> list[str]:
    """Return deduplicated list of subscriber email addresses."""
    # Check runtime cache first (updated via /api/subscribers)
    cache_file = Path("/tmp/stock_trader_subscribers.json")
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass

    raw = os.getenv("EMAIL_SUBSCRIBERS", "").strip()
    if not raw:
        # Backward compat: fall back to EMAIL_TO if EMAIL_SUBSCRIBERS not set
        raw = os.getenv("EMAIL_TO", DEFAULT_SUBSCRIBERS).strip()
    addresses = [addr.strip() for addr in raw.split(",") if addr.strip()]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for addr in addresses:
        lower = addr.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(addr)
    return result


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


def _fmt_money(value: float) -> str:
    return f"${float(value):,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _source_badge(name: str, status: str) -> str:
    ok = status.startswith("ok")
    color = "#0f766e" if ok else "#b45309"
    bg = "#ccfbf1" if ok else "#fef3c7"
    return (
        f"<span style='display:inline-block;margin:4px 6px 4px 0;padding:5px 9px;"
        f"border-radius:999px;background:{bg};color:{color};font-size:12px;"
        f"font-weight:700'>{escape(name)}: {escape(status)}</span>"
    )


def _trade_rows_html(rows: list[dict]) -> str:
    if not rows:
        return (
            "<div style='padding:14px;border:1px solid #e5e7eb;border-radius:8px;"
            "background:#f9fafb;color:#374151'>No buy or sell intents today.</div>"
        )

    body = []
    for row in rows:
        action = row["action"]
        color = "#166534" if action == "BUY" else "#991b1b"
        detail = (
            f"stop {_fmt_money(row.get('stop', 0))} / target {_fmt_money(row.get('target', 0))}"
            if action == "BUY"
            else escape(row.get("reason", ""))
        )
        body.append(
            "<tr>"
            f"<td style='padding:10px;font-weight:800;color:{color}'>{action}</td>"
            f"<td style='padding:10px;font-weight:700'>{escape(row['ticker'])}</td>"
            f"<td style='padding:10px;text-align:right'>{int(row['shares']):,}</td>"
            f"<td style='padding:10px;text-align:right'>{_fmt_money(row['price'])}</td>"
            f"<td style='padding:10px;text-align:right'>{_fmt_pct(row.get('prob_buy', 0))}</td>"
            f"<td style='padding:10px'>{detail}</td>"
            "</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:14px'>"
        "<thead><tr style='background:#f3f4f6;color:#374151'>"
        "<th style='padding:10px;text-align:left'>Action</th>"
        "<th style='padding:10px;text-align:left'>Ticker</th>"
        "<th style='padding:10px;text-align:right'>Shares</th>"
        "<th style='padding:10px;text-align:right'>Price</th>"
        "<th style='padding:10px;text-align:right'>Score</th>"
        "<th style='padding:10px;text-align:left'>Notes</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _top_scores_html(top: pd.DataFrame) -> str:
    if top.empty:
        return "<p>No model scores were available.</p>"

    rows = []
    for _, row in top.iterrows():
        rows.append(
            "<tr>"
            f"<td style='padding:9px;font-weight:800'>{escape(str(row['ticker']))}</td>"
            f"<td style='padding:9px;text-align:right'>{_fmt_money(row['price'])}</td>"
            f"<td style='padding:9px;text-align:right'>{_fmt_pct(row['base_prob_buy'])}</td>"
            f"<td style='padding:9px;text-align:right;color:#7c3aed'>{_fmt_pct(row['sentiment_adjustment'])}</td>"
            f"<td style='padding:9px;text-align:right;font-weight:800'>{_fmt_pct(row['prob_buy'])}</td>"
            f"<td style='padding:9px;text-align:right'>{int(row.get('x_mentions', 0))}</td>"
            f"<td style='padding:9px;text-align:right'>{int(row.get('reddit_mentions', 0))}</td>"
            f"<td style='padding:9px;text-align:right'>{int(row.get('news_mentions', 0))}</td>"
            "</tr>"
        )

    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        "<thead><tr style='background:#111827;color:white'>"
        "<th style='padding:10px;text-align:left'>Ticker</th>"
        "<th style='padding:10px;text-align:right'>Price</th>"
        "<th style='padding:10px;text-align:right'>Base</th>"
        "<th style='padding:10px;text-align:right'>Overlay</th>"
        "<th style='padding:10px;text-align:right'>Final</th>"
        "<th style='padding:10px;text-align:right'>X</th>"
        "<th style='padding:10px;text-align:right'>Reddit</th>"
        "<th style='padding:10px;text-align:right'>Google</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _build_html_email(
    now_ny: datetime,
    decisions: dict,
    trade_rows: list[dict],
    top: pd.DataFrame,
    source_status: dict,
) -> str:
    source_html = "".join(
        _source_badge(name, status)
        for name, status in source_status.items()
    )
    return f"""\
<!doctype html>
<html>
  <body style="margin:0;background:#f3f4f6;font-family:Arial,Helvetica,sans-serif;color:#111827">
    <div style="max-width:860px;margin:0 auto;padding:24px">
      <div style="background:#111827;color:white;padding:22px 24px;border-radius:12px 12px 0 0">
        <div style="font-size:13px;color:#cbd5e1">Stock Trader Market-Open Intent Log</div>
        <h1 style="margin:6px 0 0;font-size:24px">{now_ny:%A, %B %d, %Y}</h1>
      </div>
      <div style="background:white;padding:22px 24px;border-radius:0 0 12px 12px;border:1px solid #e5e7eb">
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px">
          <div style="padding:12px 14px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
            <div style="font-size:12px;color:#6b7280">Portfolio Value</div>
            <div style="font-size:18px;font-weight:800">{_fmt_money(decisions['portfolio_value'])}</div>
          </div>
          <div style="padding:12px 14px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
            <div style="font-size:12px;color:#6b7280">Cash</div>
            <div style="font-size:18px;font-weight:800">{_fmt_money(decisions['cash'])}</div>
          </div>
          <div style="padding:12px 14px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
            <div style="font-size:12px;color:#6b7280">Drawdown</div>
            <div style="font-size:18px;font-weight:800">{_fmt_pct(decisions['drawdown'])}</div>
          </div>
        </div>

        <h2 style="font-size:18px;margin:20px 0 10px">Trade Intents</h2>
        {_trade_rows_html(trade_rows)}

        <h2 style="font-size:18px;margin:24px 0 10px">Data Sources</h2>
        <div>{source_html}</div>

        <h2 style="font-size:18px;margin:24px 0 10px">Top Adjusted Scores</h2>
        {_top_scores_html(top)}

        <p style="margin-top:18px;color:#6b7280;font-size:13px;line-height:1.5">
          Final score = base ML probability plus a capped news/social attention overlay.
          This is an intent log only; it does not place orders.
        </p>
      </div>
    </div>
  </body>
</html>
"""


def _send_email(subject: str, body: str, html_body: str | None = None):
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    subscribers = _get_subscribers()
    email_from = os.getenv("EMAIL_FROM", smtp_user or subscribers[0]).strip()

    if not smtp_host or not smtp_user or not smtp_password:
        raise RuntimeError(
            "Missing SMTP_HOST, SMTP_USER, or SMTP_PASSWORD environment variables."
        )

    if not subscribers:
        raise RuntimeError("No email subscribers configured.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = ", ".join(subscribers)
    msg.set_content(body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)


def run_market_open_job(send_email: bool = True) -> dict:
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    bundle = _load_model_bundle()
    tickers = bundle["metadata"]["tickers"]

    base_scored_df = score_today(tickers, model_bundle=bundle)
    if base_scored_df.empty:
        raise RuntimeError("Model scoring returned no rows.")
    scored_df, overlay_df = apply_sentiment_overlay(base_scored_df)

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

    top_cols = [
        "ticker", "price", "base_prob_buy", "sentiment_adjustment", "prob_buy",
        "x_mentions", "reddit_mentions", "news_mentions", "attention_score",
        "rsi", "bb_pct",
    ]
    top = scored_df.head(10)[[c for c in top_cols if c in scored_df.columns]]
    source_status = scored_df.attrs.get("source_status", {})
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
        "prob_buy includes the capped news/Reddit attention overlay.",
        "base_prob_buy is the original ML score before the overlay.",
        "",
        "This email is an intent log only; it does not place orders.",
    ])
    html_body = _build_html_email(
        now_ny=now_ny,
        decisions=decisions,
        trade_rows=trade_rows,
        top=top,
        source_status=source_status,
    )

    log_entry = {
        "date": now_ny.isoformat(),
        "trade_rows": trade_rows,
        "decisions": {
            "portfolio_value": decisions["portfolio_value"],
            "cash": decisions["cash"],
            "drawdown": decisions["drawdown"],
        },
        "top_scores": top.to_dict(orient="records"),
        "source_status": source_status,
        "sentiment_overlay": overlay_df.head(25).to_dict(orient="records"),
        "positions": {
            ticker: asdict(position)
            for ticker, position in portfolio.positions.items()
        },
    }
    log_file = LOG_DIR / f"market_open_intents_{now_ny:%Y%m%d_%H%M%S}.json"
    log_file.write_text(json.dumps(log_entry, indent=2, default=str))

    if send_email:
        _send_email(subject, body, html_body=html_body)

    return {
        "ok": True,
        "date": now_ny.isoformat(),
        "email_sent": send_email,
        "subscribers": _get_subscribers() if send_email else [],
        "trade_count": len(trade_rows),
        "buy_count": sum(1 for row in trade_rows if row["action"] == "BUY"),
        "sell_count": sum(1 for row in trade_rows if row["action"] == "SELL"),
        "log_file": str(log_file),
        "trades": trade_rows,
    }

# Vercel Market-Open Email Cron

This project can run a Vercel Cron Job that emails the model's daily buy/sell
intents at market open. It does not place trades.

## Endpoint

Vercel calls:

```text
GET /api/market_open
```

You can test without sending email:

```text
GET /api/market_open?dry_run=true
```

If `CRON_SECRET` is set, pass either:

```text
Authorization: Bearer <CRON_SECRET>
```

or:

```text
GET /api/market_open?secret=<CRON_SECRET>
```

## Schedule

`vercel.json` schedules both:

```text
30 13 * * 1-5
30 14 * * 1-5
```

Vercel cron uses UTC. The endpoint only runs during the New York market-open
window, so one schedule covers daylight time and the other covers standard time.

## Required Environment Variables

Set these in Vercel Project Settings -> Environment Variables:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=<your sending email>
SMTP_PASSWORD=<your app password>
EMAIL_TO=bryan.g.shi@gmail.com
EMAIL_FROM=<your sending email>
CRON_SECRET=<random long secret>
```

For Gmail, `SMTP_PASSWORD` should be an app password, not your normal account
password.

## Optional Portfolio State

Without a portfolio state, the cron starts from the default paper portfolio:

```json
{"cash": 100000, "peak_value": 100000, "positions": {}}
```

To allow sell recommendations for existing holdings, set `PORTFOLIO_STATE_JSON`
to a JSON object with the same shape as `.portfolio_state.json`, for example:

```json
{
  "cash": 25000,
  "peak_value": 100000,
  "positions": {
    "AAPL": {
      "ticker": "AAPL",
      "shares": 10,
      "avg_cost": 190.0,
      "entry_date": "2026-06-01",
      "stop_loss": 175.0,
      "target_price": 220.0,
      "entry_score": 0.65
    }
  }
}
```

## Deploy Notes

Deploy from the `stock_trader` directory so Vercel sees:

- `api/market_open.py`
- `vercel.json`
- `requirements.txt`
- `models/rf_latest.pkl`
- `models/gb_latest.pkl`
- `models/lr_latest.pkl`
- `models/metadata.json`

The function writes runtime JSON logs to `/tmp/stock_trader_logs` on Vercel.
The email is the durable daily log unless you add external storage later.

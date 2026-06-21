# Vercel Market-Open Email Cron

This project can run a Vercel Cron Job that emails the model's daily buy/sell
intents before market open every calendar day. It does not place trades.

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

Vercel cron uses UTC. `vercel.json` schedules both:

```text
30 13 * * *
30 14 * * *
```

The endpoint only runs during the New York premarket email window, including
weekends and market holidays, so one schedule covers daylight time and the other
covers standard time.

## Required Environment Variables

Set these in Vercel Project Settings -> Environment Variables:

```text
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=<your sending email>
SMTP_PASSWORD=<your app password>
EMAIL_SUBSCRIBERS=bryan.g.shi@gmail.com
EMAIL_FROM=<your sending email>
CRON_SECRET=<random long secret>
```

For Gmail, `SMTP_PASSWORD` should be an app password, not your normal account
password.

### Email Subscribers

The system supports multiple subscribers. Set `EMAIL_SUBSCRIBERS` to a
comma-separated list of email addresses:

```text
EMAIL_SUBSCRIBERS=bryan.g.shi@gmail.com,friend@example.com,colleague@example.com
```

For backward compatibility, if `EMAIL_SUBSCRIBERS` is not set, the system
falls back to `EMAIL_TO`, then to `bryan.g.shi@gmail.com`.

You can also manage subscribers at runtime via the API:

```text
GET  /api/subscribers                    → list subscribers
POST /api/subscribers?email=new@test.com → add subscriber (requires CRON_SECRET)
DELETE /api/subscribers?email=old@test.com → remove subscriber (requires CRON_SECRET)
```

Note: Runtime subscriber changes persist in `/tmp` which resets on cold starts.
For permanent changes, update `EMAIL_SUBSCRIBERS` in Vercel env vars.

## Optional Sentiment Overlay Variables

The market-open email applies a capped news/Reddit attention overlay to the
base model score. These are optional:

```text
SENTIMENT_MAX_ADJUST=0.12
NEWS_TOP_N=12
REDDIT_SUBREDDITS=wallstreetbets+investing+stocks+news
REDDIT_LISTINGS=hot,new,rising,top
REDDIT_TOP_TIME_FILTER=day
REDDIT_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36
REDDIT_COOKIE=<optional reddit browser Cookie header>
X_TOP_N=12
X_SEARCH_PAGES=2
X_BEARER_TOKEN=<X/Twitter API v2 Bearer token - primary method>
X_AUTH_TOKEN=<optional x auth_token cookie - fallback for Playwright scrape>
X_CT0=<optional x ct0 cookie - fallback for Playwright scrape>
PLAYWRIGHT_CDP_URL=<browserless/browserbase cdp websocket url>
PLAYWRIGHT_WS_ENDPOINT=<optional playwright websocket endpoint>
X_RUNTIME_BROWSER_INSTALL=false
SENTIMENT_REQUEST_TIMEOUT=6
```

### X/Twitter Data Source

The system uses a two-tier approach for X data:

1. **Primary: X API v2** — Uses `X_BEARER_TOKEN` to call the Twitter/X recent
   search API. This is fast, reliable, and works on Vercel without Playwright.
   Get a bearer token from the [X Developer Portal](https://developer.x.com).

2. **Fallback: Playwright scraping** — If the API fails (auth error, rate limit),
   falls back to scraping X search pages via a remote browser. Requires
   `PLAYWRIGHT_CDP_URL` and optionally `X_AUTH_TOKEN` + `X_CT0` cookies.

### Reddit Data Source

Reddit JSON endpoints (`/r/stocks.json`) work locally but are blocked from
Vercel server IPs. The system handles this automatically:

1. **Direct fetch** — Tries the public JSON endpoint first (works locally).
2. **Remote browser fallback** — When the direct fetch fails (403 on Vercel),
   automatically retries through the remote browser at `PLAYWRIGHT_CDP_URL`.
   The remote browser fetches the JSON endpoint from a residential/cloud IP
   that Reddit doesn't block.

If Reddit still returns a login/block page through the remote browser, set
`REDDIT_COOKIE` to the full `Cookie` request header copied from a logged-in
browser request to `https://www.reddit.com/r/stocks.json`.

### Remote Browser (Browserless)

Both Reddit fallback and X Playwright scraping use the same remote browser
endpoint. Set `PLAYWRIGHT_CDP_URL` to your Browserless WebSocket URL:

```text
PLAYWRIGHT_CDP_URL=wss://chrome.browserless.io?token=YOUR_TOKEN
```

This single endpoint handles both data sources on Vercel.

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
- `api/subscribers.py`
- `vercel.json`
- `requirements.txt`
- `models/rf_latest.pkl`
- `models/gb_latest.pkl`
- `models/lr_latest.pkl`
- `models/metadata.json`

The function writes runtime JSON logs to `/tmp/stock_trader_logs` on Vercel.
The email is the durable daily log unless you add external storage later.

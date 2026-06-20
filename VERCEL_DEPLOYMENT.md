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

## Optional Sentiment Overlay Variables

The market-open email applies a capped news/Reddit attention overlay to the
base model score. These are optional:

```text
SENTIMENT_MAX_ADJUST=0.12
NEWS_TOP_N=12
REDDIT_SUBREDDITS=wallstreetbets+investing+stocks+news
REDDIT_LISTINGS=hot,new,rising,top
REDDIT_TOP_TIME_FILTER=day
REDDIT_USER_AGENT=windows:stock-trader-market-open:v1.0 (by /u/BarnyIsh)
X_TOP_N=12
X_SEARCH_PAGES=2
X_AUTH_TOKEN=<optional x auth_token cookie>
X_CT0=<optional x ct0 cookie>
PLAYWRIGHT_CDP_URL=<optional browserless/browserbase cdp websocket url>
PLAYWRIGHT_WS_ENDPOINT=<optional playwright websocket endpoint>
X_RUNTIME_BROWSER_INSTALL=false
SENTIMENT_REQUEST_TIMEOUT=6
```

`prob_buy` in the email is the adjusted score. `base_prob_buy` is the original
ML score before the overlay. Reddit is fetched from public subreddit JSON
endpoints such as `https://www.reddit.com/r/stocks.json`; no Reddit API
credentials are required. X is scraped with Playwright from search pages, so no
X API key is required. X may
redirect anonymous headless browsers to login; if that happens, set the
optional `X_AUTH_TOKEN` and `X_CT0` cookie values from a browser session.
Playwright needs a browser binary such as Chromium. Vercel's Python runtime can
bundle the browser with extended function limits, but the runtime still lacks
some shared Linux libraries needed to launch it. The reliable Vercel setup is a
remote browser service. Set `PLAYWRIGHT_CDP_URL` for a Chrome DevTools Protocol
endpoint, or `PLAYWRIGHT_WS_ENDPOINT` for a Playwright protocol endpoint. The X
scraper still uses Playwright; it just connects to a browser running outside
Vercel.

Reddit is fetched from public JSON endpoints. These URLs may open in your
browser but still return 403 from Vercel because Reddit blocks some server IPs.
When a remote browser endpoint is configured, Reddit JSON fetches that are
blocked from Vercel are retried through the same remote browser.
Facebook/Meta public post search is not included by default because useful
public content access requires approved Meta Graph API permissions.

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

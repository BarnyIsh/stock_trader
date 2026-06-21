"""
api/subscribe.py - User-friendly subscription page.

GET  /api/subscribe          → renders the subscribe/unsubscribe HTML form
POST /api/subscribe          → handles form submission (subscribe or unsubscribe)

No CRON_SECRET required to subscribe (public form).
Unsubscribe requires the email to already exist in the list.
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_SUBSCRIBERS = "bryan.g.shi@gmail.com"
SUBSCRIBERS_FILE = Path("/tmp/stock_trader_subscribers.json")


def _load_subscribers() -> list[str]:
    if SUBSCRIBERS_FILE.exists():
        try:
            data = json.loads(SUBSCRIBERS_FILE.read_text())
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    raw = os.getenv("EMAIL_SUBSCRIBERS", "").strip()
    if not raw:
        raw = os.getenv("EMAIL_TO", DEFAULT_SUBSCRIBERS).strip()
    addresses = [addr.strip() for addr in raw.split(",") if addr.strip()]
    seen = set()
    result = []
    for addr in addresses:
        lower = addr.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(addr)
    return result


def _save_subscribers(subscribers: list[str]):
    SUBSCRIBERS_FILE.write_text(json.dumps(subscribers))
    os.environ["EMAIL_SUBSCRIBERS"] = ",".join(subscribers)


def _html_response(handler, status: int, body: str):
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _render_page(message: str = "", error: str = "") -> str:
    msg_html = ""
    if message:
        msg_html = (
            f'<div style="background:#ccfbf1;border:1px solid #0f766e;color:#0f766e;'
            f'padding:14px 18px;border-radius:8px;margin-bottom:20px;font-weight:500">'
            f'{message}</div>'
        )
    if error:
        msg_html = (
            f'<div style="background:#fee2e2;border:1px solid #991b1b;color:#991b1b;'
            f'padding:14px 18px;border-radius:8px;margin-bottom:20px;font-weight:500">'
            f'{error}</div>'
        )

    return f"""\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stock Trader — Email Subscription</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f3f4f6;
      color: #111827;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .card {{
      background: white;
      border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 40px;
      max-width: 460px;
      width: 100%;
    }}
    .header {{
      text-align: center;
      margin-bottom: 28px;
    }}
    .header h1 {{
      font-size: 24px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .header p {{
      color: #6b7280;
      font-size: 15px;
      line-height: 1.5;
    }}
    .form-group {{
      margin-bottom: 16px;
    }}
    label {{
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: #374151;
      margin-bottom: 6px;
    }}
    input[type="email"] {{
      width: 100%;
      padding: 12px 16px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      font-size: 15px;
      transition: border-color 0.2s;
      outline: none;
    }}
    input[type="email"]:focus {{
      border-color: #111827;
      box-shadow: 0 0 0 3px rgba(17,24,39,0.08);
    }}
    .btn-row {{
      display: flex;
      gap: 10px;
      margin-top: 20px;
    }}
    button {{
      flex: 1;
      padding: 12px 20px;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-subscribe {{
      background: #111827;
      color: white;
    }}
    .btn-subscribe:hover {{
      background: #374151;
    }}
    .btn-unsubscribe {{
      background: #f3f4f6;
      color: #374151;
      border: 1px solid #d1d5db;
    }}
    .btn-unsubscribe:hover {{
      background: #e5e7eb;
    }}
    .footer {{
      text-align: center;
      margin-top: 24px;
      color: #9ca3af;
      font-size: 12px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1>Stock Trader Alerts</h1>
      <p>Get daily pre-market trade intent emails with ML scores, sentiment overlay, and portfolio recommendations.</p>
    </div>

    {msg_html}

    <form method="POST" action="/api/subscribe">
      <div class="form-group">
        <label for="email">Email address</label>
        <input type="email" id="email" name="email" required
               placeholder="you@example.com">
      </div>
      <div class="btn-row">
        <button type="submit" name="action" value="subscribe" class="btn-subscribe">
          Subscribe
        </button>
        <button type="submit" name="action" value="unsubscribe" class="btn-unsubscribe">
          Unsubscribe
        </button>
      </div>
    </form>

    <div class="footer">
      Emails are sent daily before market open (9:30 AM ET).<br>
      You can unsubscribe at any time using this page.
    </div>
  </div>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        _html_response(self, 200, _render_page())

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        params = parse_qs(body)

        email = params.get("email", [""])[0].strip().lower()
        action = params.get("action", ["subscribe"])[0].strip()

        if not email or "@" not in email or "." not in email.split("@")[-1]:
            _html_response(self, 400, _render_page(error="Please enter a valid email address."))
            return

        subscribers = _load_subscribers()
        existing = {s.lower() for s in subscribers}

        if action == "subscribe":
            if email in existing:
                _html_response(self, 200, _render_page(
                    message=f"{email} is already subscribed! You'll get the next market-open email."
                ))
                return
            subscribers.append(email)
            _save_subscribers(subscribers)
            _html_response(self, 200, _render_page(
                message=f"Subscribed! {email} will receive daily pre-market alerts."
            ))

        elif action == "unsubscribe":
            if email not in existing:
                _html_response(self, 200, _render_page(
                    error=f"{email} is not currently subscribed."
                ))
                return
            filtered = [s for s in subscribers if s.lower() != email]
            if not filtered:
                _html_response(self, 400, _render_page(
                    error="Cannot remove the last subscriber."
                ))
                return
            _save_subscribers(filtered)
            _html_response(self, 200, _render_page(
                message=f"Unsubscribed. {email} will no longer receive alerts."
            ))

        else:
            _html_response(self, 400, _render_page(error="Invalid action."))

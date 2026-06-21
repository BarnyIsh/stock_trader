"""
api/subscribers.py - Manage email subscriber list.

GET  /api/subscribers              → list current subscribers
POST /api/subscribers?email=X      → add a subscriber
DELETE /api/subscribers?email=X    → remove a subscriber

All mutations require CRON_SECRET for authorization.
Subscribers are stored in the EMAIL_SUBSCRIBERS env var (comma-separated).
On Vercel, this endpoint reads/writes a JSON file in /tmp as a runtime cache.
For persistent changes, update the EMAIL_SUBSCRIBERS env var in Vercel settings.
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_SUBSCRIBERS = "bryan.g.shi@gmail.com"
SUBSCRIBERS_FILE = Path("/tmp/stock_trader_subscribers.json")


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict):
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _authorized(handler: BaseHTTPRequestHandler, query: dict) -> bool:
    secret = os.getenv("CRON_SECRET", "").strip()
    if not secret:
        return True
    auth = handler.headers.get("Authorization", "")
    query_secret = query.get("secret", [""])[0]
    return auth == f"Bearer {secret}" or query_secret == secret


def _load_subscribers() -> list[str]:
    """Load subscribers from file cache, falling back to env var."""
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
    """Persist subscriber list to /tmp cache and update env for current process."""
    SUBSCRIBERS_FILE.write_text(json.dumps(subscribers))
    os.environ["EMAIL_SUBSCRIBERS"] = ",".join(subscribers)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        subscribers = _load_subscribers()
        _json_response(self, 200, {
            "ok": True,
            "subscribers": subscribers,
            "count": len(subscribers),
        })

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if not _authorized(self, query):
            _json_response(self, 401, {"ok": False, "error": "unauthorized"})
            return

        email = query.get("email", [""])[0].strip().lower()
        if not email or "@" not in email:
            _json_response(self, 400, {"ok": False, "error": "invalid email"})
            return

        subscribers = _load_subscribers()
        existing = {s.lower() for s in subscribers}
        if email in existing:
            _json_response(self, 200, {
                "ok": True,
                "message": "already subscribed",
                "subscribers": subscribers,
            })
            return

        subscribers.append(email)
        _save_subscribers(subscribers)
        _json_response(self, 200, {
            "ok": True,
            "message": f"added {email}",
            "subscribers": subscribers,
        })

    def do_DELETE(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if not _authorized(self, query):
            _json_response(self, 401, {"ok": False, "error": "unauthorized"})
            return

        email = query.get("email", [""])[0].strip().lower()
        if not email:
            _json_response(self, 400, {"ok": False, "error": "email required"})
            return

        subscribers = _load_subscribers()
        filtered = [s for s in subscribers if s.lower() != email]

        if len(filtered) == len(subscribers):
            _json_response(self, 404, {
                "ok": False,
                "error": f"{email} not found in subscribers",
            })
            return

        if not filtered:
            _json_response(self, 400, {
                "ok": False,
                "error": "cannot remove last subscriber",
            })
            return

        _save_subscribers(filtered)
        _json_response(self, 200, {
            "ok": True,
            "message": f"removed {email}",
            "subscribers": filtered,
        })

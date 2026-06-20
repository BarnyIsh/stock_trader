import json
import os
from datetime import datetime, time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from market_open_job import run_market_open_job


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


def _market_open_window() -> bool:
    if os.getenv("DISABLE_MARKET_OPEN_TIME_GUARD", "").lower() == "true":
        return True
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    return time(9, 25) <= now.time() <= time(10, 5)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if not _authorized(self, query):
            _json_response(self, 401, {"ok": False, "error": "unauthorized"})
            return

        if query.get("dry_run", [""])[0].lower() == "true":
            try:
                result = run_market_open_job(send_email=False)
                _json_response(self, 200, result)
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        if not _market_open_window():
            _json_response(self, 200, {
                "ok": True,
                "skipped": True,
                "reason": "outside_market_open_window",
            })
            return

        try:
            result = run_market_open_job(send_email=True)
            _json_response(self, 200, result)
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})

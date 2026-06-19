"""
schwab_client.py - Schwab Developer API OAuth2 client + order execution

OAuth2 flow (Authorization Code):
  1. Call `get_auth_url()` → open in browser → user logs in
  2. Schwab redirects to callback_url?code=XXX
  3. Call `exchange_code(code)` → stores tokens
  4. All subsequent requests auto-refresh the access token.

Docs: https://developer.schwab.com/products/trader-api--individual-
"""

import json
import time
import base64
import requests
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs
from config import (
    SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL,
    SCHWAB_AUTH_URL, SCHWAB_TOKEN_URL, SCHWAB_BASE_URL, SCHWAB_MARKET_URL,
    PAPER_TRADING
)

TOKEN_FILE = Path(__file__).parent / ".schwab_tokens.json"


class SchwabClient:
    """
    Thin wrapper around the Schwab Individual Trader API.
    Handles OAuth2, token refresh, quotes, account info, and order placement.
    """

    def __init__(self):
        self.app_key    = SCHWAB_APP_KEY
        self.app_secret = SCHWAB_APP_SECRET
        self.tokens: dict = {}
        self._load_tokens()

    # ── Auth helpers ─────────────────────────────────────────────────────────

    def get_auth_url(self) -> str:
        """Step 1: Return the URL the user must visit to authorize."""
        params = {
            "response_type": "code",
            "client_id":     self.app_key,
            "redirect_uri":  SCHWAB_CALLBACK_URL,
            "scope":         "readonly trading",
        }
        return f"{SCHWAB_AUTH_URL}?{urlencode(params)}"

    def exchange_code(self, code: str) -> dict:
        """Step 2: Exchange authorization code for access + refresh tokens."""
        creds = base64.b64encode(
            f"{self.app_key}:{self.app_secret}".encode()
        ).decode()
        resp = requests.post(
            SCHWAB_TOKEN_URL,
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={
                "grant_type":   "authorization_code",
                "code":         code,
                "redirect_uri": SCHWAB_CALLBACK_URL,
            },
        )
        resp.raise_for_status()
        self.tokens = resp.json()
        self.tokens["expires_at"] = time.time() + self.tokens.get("expires_in", 1800)
        self._save_tokens()
        return self.tokens

    def _refresh_access_token(self):
        creds = base64.b64encode(
            f"{self.app_key}:{self.app_secret}".encode()
        ).decode()
        resp = requests.post(
            SCHWAB_TOKEN_URL,
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={
                "grant_type":    "refresh_token",
                "refresh_token": self.tokens["refresh_token"],
            },
        )
        resp.raise_for_status()
        self.tokens.update(resp.json())
        self.tokens["expires_at"] = time.time() + self.tokens.get("expires_in", 1800)
        self._save_tokens()

    def _get_headers(self) -> dict:
        if not self.tokens:
            raise RuntimeError(
                "Not authenticated. Call get_auth_url() → exchange_code() first."
            )
        if time.time() >= self.tokens.get("expires_at", 0) - 60:
            self._refresh_access_token()
        return {
            "Authorization": f"Bearer {self.tokens['access_token']}",
            "Content-Type":  "application/json",
        }

    def _save_tokens(self):
        TOKEN_FILE.write_text(json.dumps(self.tokens, indent=2))

    def _load_tokens(self):
        if TOKEN_FILE.exists():
            self.tokens = json.loads(TOKEN_FILE.read_text())

    # ── Account ──────────────────────────────────────────────────────────────

    def get_accounts(self) -> list:
        """Return list of linked Schwab accounts with balances."""
        r = requests.get(
            f"{SCHWAB_BASE_URL}/accounts",
            headers=self._get_headers(),
            params={"fields": "positions"},
        )
        r.raise_for_status()
        return r.json()

    def get_account_number(self) -> str:
        """Return the first account's encrypted account number."""
        accs = self.get_accounts()
        return accs[0]["hashValue"]

    def get_portfolio_value(self) -> float:
        """Total liquidation value of the first account."""
        accs = self.get_accounts()
        return accs[0]["securitiesAccount"]["currentBalances"]["liquidationValue"]

    def get_positions(self) -> list:
        """Return current open positions."""
        accs = self.get_accounts()
        return accs[0]["securitiesAccount"].get("positions", [])

    # ── Market Data ──────────────────────────────────────────────────────────

    def get_quotes(self, symbols: list[str]) -> dict:
        """
        Fetch real-time quotes for a list of symbols.
        Returns dict keyed by symbol.
        """
        r = requests.get(
            f"{SCHWAB_MARKET_URL}/quotes",
            headers=self._get_headers(),
            params={"symbols": ",".join(symbols), "fields": "quote,fundamental"},
        )
        r.raise_for_status()
        return r.json()

    def get_price_history(
        self,
        symbol: str,
        period_type: str = "year",
        period: int = 5,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> dict:
        """Fetch OHLCV history for a symbol."""
        r = requests.get(
            f"{SCHWAB_MARKET_URL}/pricehistory",
            headers=self._get_headers(),
            params={
                "symbol":        symbol,
                "periodType":    period_type,
                "period":        period,
                "frequencyType": frequency_type,
                "frequency":     frequency,
            },
        )
        r.raise_for_status()
        return r.json()

    def search_instruments(self, query: str, projection: str = "symbol-search") -> dict:
        """Screen / search instruments by name or symbol."""
        r = requests.get(
            f"{SCHWAB_MARKET_URL}/instruments",
            headers=self._get_headers(),
            params={"symbol": query, "projection": projection},
        )
        r.raise_for_status()
        return r.json()

    def get_movers(self, index: str = "$DJI", sort: str = "PERCENT_CHANGE_UP") -> dict:
        """Get top movers for a given index."""
        r = requests.get(
            f"{SCHWAB_MARKET_URL}/movers/{index}",
            headers=self._get_headers(),
            params={"sort": sort, "frequency": 1},
        )
        r.raise_for_status()
        return r.json()

    # ── Orders ───────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        action: str = "BUY",          # "BUY" | "SELL"
        account_number: str | None = None,
    ) -> dict:
        """
        Place a market order.
        In PAPER_TRADING mode prints the order without sending it.
        """
        account_number = account_number or self.get_account_number()
        order = {
            "orderType":   "MARKET",
            "session":     "NORMAL",
            "duration":    "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": action,
                    "quantity":    quantity,
                    "instrument":  {
                        "symbol":    symbol,
                        "assetType": "EQUITY",
                    },
                }
            ],
        }
        if PAPER_TRADING:
            print(f"[PAPER] {action} {quantity} × {symbol} @ MARKET")
            return {"status": "paper", "order": order}

        r = requests.post(
            f"{SCHWAB_BASE_URL}/accounts/{account_number}/orders",
            headers=self._get_headers(),
            json=order,
        )
        r.raise_for_status()
        return {"status": "submitted", "location": r.headers.get("Location", "")}

    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        limit_price: float,
        action: str = "BUY",
        account_number: str | None = None,
    ) -> dict:
        """Place a limit order (preferred for production use)."""
        account_number = account_number or self.get_account_number()
        order = {
            "orderType":   "LIMIT",
            "session":     "NORMAL",
            "duration":    "DAY",
            "price":       str(round(limit_price, 2)),
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": action,
                    "quantity":    quantity,
                    "instrument":  {
                        "symbol":    symbol,
                        "assetType": "EQUITY",
                    },
                }
            ],
        }
        if PAPER_TRADING:
            print(f"[PAPER] {action} {quantity} × {symbol} LIMIT @ ${limit_price:.2f}")
            return {"status": "paper", "order": order}

        r = requests.post(
            f"{SCHWAB_BASE_URL}/accounts/{account_number}/orders",
            headers=self._get_headers(),
            json=order,
        )
        r.raise_for_status()
        return {"status": "submitted", "location": r.headers.get("Location", "")}

    def cancel_order(self, order_id: str, account_number: str | None = None) -> bool:
        account_number = account_number or self.get_account_number()
        r = requests.delete(
            f"{SCHWAB_BASE_URL}/accounts/{account_number}/orders/{order_id}",
            headers=self._get_headers(),
        )
        return r.status_code == 200

    def get_orders(self, account_number: str | None = None) -> list:
        account_number = account_number or self.get_account_number()
        r = requests.get(
            f"{SCHWAB_BASE_URL}/accounts/{account_number}/orders",
            headers=self._get_headers(),
        )
        r.raise_for_status()
        return r.json()

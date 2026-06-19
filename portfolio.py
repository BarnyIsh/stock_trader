"""
portfolio.py - Position sizing, risk management, and portfolio rebalancing

Implements:
  - Kelly-fraction position sizing (capped)
  - ATR-based stop-loss placement
  - Max drawdown circuit breaker
  - Concentration limits
  - Daily buy/sell decision logic
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib     import Path
from datetime    import datetime
from config import (
    MAX_POSITION_SIZE, MAX_PORTFOLIO_STOCKS,
    RISK_PER_TRADE, BUY_SCORE_THRESHOLD, SELL_SCORE_THRESHOLD,
    PAPER_TRADING
)

STATE_FILE = Path(__file__).parent / ".portfolio_state.json"


@dataclass
class Position:
    ticker:       str
    shares:       int
    avg_cost:     float
    entry_date:   str
    stop_loss:    float
    target_price: float
    entry_score:  float

    @property
    def current_value(self) -> float:
        return self.shares * self.avg_cost   # updated externally

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Portfolio:
    cash:         float
    peak_value:   float
    positions:    dict = field(default_factory=dict)   # ticker → Position
    trade_log:    list = field(default_factory=list)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self):
        state = {
            "cash":       self.cash,
            "peak_value": self.peak_value,
            "positions":  {k: v.to_dict() for k, v in self.positions.items()},
            "trade_log":  self.trade_log[-500:],   # keep last 500
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls) -> "Portfolio":
        if STATE_FILE.exists():
            s = json.loads(STATE_FILE.read_text())
            p = cls(cash=s["cash"], peak_value=s["peak_value"])
            p.positions = {
                k: Position(**v) for k, v in s.get("positions", {}).items()
            }
            p.trade_log = s.get("trade_log", [])
            return p
        return cls(cash=100_000.0, peak_value=100_000.0)

    # ── Accounting ────────────────────────────────────────────────────────────

    def total_value(self, prices: dict[str, float]) -> float:
        equity = sum(
            pos.shares * prices.get(t, pos.avg_cost)
            for t, pos in self.positions.items()
        )
        return self.cash + equity

    def update_peak(self, total_val: float):
        if total_val > self.peak_value:
            self.peak_value = total_val

    def current_drawdown(self, total_val: float) -> float:
        return (self.peak_value - total_val) / self.peak_value

    # ── Position sizing (fractional Kelly, ATR-based stop) ────────────────────

    def size_position(
        self,
        ticker: str,
        price: float,
        atr: float,
        prob_buy: float,
        total_val: float,
    ) -> tuple[int, float, float]:
        """
        Returns (shares, stop_price, target_price).

        Kelly fraction: f* = (p*b - q) / b  where b = risk/reward
        Capped at MAX_POSITION_SIZE of portfolio.
        """
        if price <= 0 or atr <= 0:
            return 0, 0.0, 0.0

        # ATR-based stop: 2 × ATR below entry
        stop_loss  = price - 2.0 * atr
        target     = price + 3.0 * atr          # 3:1 reward-risk

        risk_per_share = price - stop_loss
        if risk_per_share <= 0:
            return 0, stop_loss, target

        # Kelly fraction (conservative half-Kelly)
        p = prob_buy
        q = 1 - p
        b = (target - price) / risk_per_share   # reward-to-risk
        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(0, kelly * 0.5)             # half-Kelly for conservatism

        # Dollar allocation: min(Kelly %, MAX_POSITION_SIZE) of portfolio
        pct    = min(kelly, MAX_POSITION_SIZE)
        alloc  = pct * total_val
        alloc  = min(alloc, self.cash * 0.95)  # don't use more than 95% cash
        shares = int(alloc / price)
        return shares, round(stop_loss, 2), round(target, 2)

    # ── Daily decision engine ─────────────────────────────────────────────────

    def decide_trades(
        self,
        scored_df:    pd.DataFrame,      # from model.score_today()
        research_df:  pd.DataFrame,      # from market_research.run_market_research()
        current_prices: dict[str, float],# live quotes
        atr_map:      dict[str, float],  # ticker → ATR
        max_drawdown_abort: float = 0.20,
    ) -> dict:
        """
        Core daily logic:
          1. Check circuit breaker (drawdown)
          2. Sell: overvalued, stop-loss hit, score degraded
          3. Buy: top-ranked undervalued candidates

        Returns {
          "buys":  [(ticker, shares, limit_price, stop, target)],
          "sells": [(ticker, shares, reason)],
        }
        """
        total_val = self.total_value(current_prices)
        self.update_peak(total_val)
        dd = self.current_drawdown(total_val)

        result = {"buys": [], "sells": [], "portfolio_value": total_val,
                  "drawdown": dd, "cash": self.cash}

        # ── Circuit breaker ──────────────────────────────────────────────────
        if dd >= max_drawdown_abort:
            print(f"⚠️  CIRCUIT BREAKER: drawdown {dd:.1%} ≥ {max_drawdown_abort:.1%}. "
                  "Selling all positions, halting buys.")
            for ticker, pos in list(self.positions.items()):
                result["sells"].append(
                    (ticker, pos.shares, "circuit_breaker_drawdown")
                )
            return result

        # Build score lookup from ML model
        score_map: dict[str, float] = {}
        if not scored_df.empty and "ticker" in scored_df.columns:
            score_map = dict(zip(scored_df["ticker"], scored_df["prob_buy"]))

        # Build research score lookup
        res_map: dict[str, float] = {}
        if not research_df.empty and "ticker" in research_df.columns:
            res_map = dict(zip(research_df["ticker"], research_df["composite_score"]))

        # ── Sell decisions ────────────────────────────────────────────────────
        for ticker, pos in list(self.positions.items()):
            price = current_prices.get(ticker, pos.avg_cost)
            ml_score  = score_map.get(ticker, 0.5)
            fun_score = res_map.get(ticker, 0.5)
            combined  = 0.6 * ml_score + 0.4 * fun_score

            # Stop-loss hit
            if price <= pos.stop_loss:
                result["sells"].append((ticker, pos.shares, "stop_loss"))
                continue

            # Price target reached
            if price >= pos.target_price:
                result["sells"].append((ticker, pos.shares, "target_reached"))
                continue

            # Model + fundamentals say sell (overvalued)
            if combined < SELL_SCORE_THRESHOLD:
                result["sells"].append((ticker, pos.shares, "score_degraded"))
                continue

        # ── Buy decisions ─────────────────────────────────────────────────────
        n_held        = len(self.positions) - len(result["sells"])
        slots_avail   = MAX_PORTFOLIO_STOCKS - max(n_held, 0)

        if slots_avail <= 0 or self.cash < 500:
            return result

        # Merge ML + research scores
        if not scored_df.empty:
            candidates = scored_df.copy()
        elif not research_df.empty:
            candidates = research_df.rename(columns={"composite_score": "prob_buy"})
        else:
            return result

        # Only consider research-validated tickers
        if not research_df.empty and "ticker" in research_df.columns:
            valid = set(research_df[
                research_df["composite_score"] >= 0.40
            ]["ticker"].tolist())
            candidates = candidates[candidates["ticker"].isin(valid)]

        # Exclude already-held positions
        held = set(self.positions.keys())
        candidates = candidates[~candidates["ticker"].isin(held)]

        # Buy threshold
        candidates = candidates[candidates["prob_buy"] >= BUY_SCORE_THRESHOLD]
        candidates = candidates.sort_values("prob_buy", ascending=False)

        for _, row in candidates.head(slots_avail).iterrows():
            ticker = row["ticker"]
            price  = current_prices.get(ticker, row.get("price", 0))
            if not price or price <= 0:
                continue
            atr = atr_map.get(ticker, price * 0.015)   # fallback 1.5% ATR

            shares, stop, target = self.size_position(
                ticker, price, atr, row["prob_buy"], total_val
            )
            if shares <= 0:
                continue

            cost = shares * price
            if cost > self.cash:
                continue

            result["buys"].append((ticker, shares, round(price, 2), stop, target))

        return result

    # ── Execute trades ────────────────────────────────────────────────────────

    def apply_sells(self, sells: list, prices: dict[str, float]):
        for ticker, shares, reason in sells:
            if ticker not in self.positions:
                continue
            pos   = self.positions.pop(ticker)
            price = prices.get(ticker, pos.avg_cost)
            pnl   = (price - pos.avg_cost) * shares
            self.cash += price * shares
            self.trade_log.append({
                "date":   datetime.now().isoformat(),
                "action": "SELL",
                "ticker": ticker,
                "shares": shares,
                "price":  price,
                "reason": reason,
                "pnl":    round(pnl, 2),
            })
            print(f"  SELL {shares:>4} × {ticker:<6} @ ${price:.2f}  "
                  f"P&L: ${pnl:+.2f}  [{reason}]")

    def apply_buys(
        self, buys: list, prices: dict[str, float], score_map: dict[str, float]
    ):
        for ticker, shares, limit, stop, target in buys:
            price = prices.get(ticker, limit)
            cost  = shares * price
            if cost > self.cash:
                print(f"  [skip] {ticker}: insufficient cash (need ${cost:.0f})")
                continue
            self.cash -= cost
            self.positions[ticker] = Position(
                ticker=ticker, shares=shares, avg_cost=price,
                entry_date=datetime.now().date().isoformat(),
                stop_loss=stop, target_price=target,
                entry_score=score_map.get(ticker, 0.0),
            )
            self.trade_log.append({
                "date":   datetime.now().isoformat(),
                "action": "BUY",
                "ticker": ticker,
                "shares": shares,
                "price":  price,
                "stop":   stop,
                "target": target,
            })
            print(f"  BUY  {shares:>4} × {ticker:<6} @ ${price:.2f}  "
                  f"stop: ${stop:.2f}  target: ${target:.2f}")

    def print_summary(self, prices: dict[str, float]):
        total = self.total_value(prices)
        print(f"\n{'─'*50}")
        print(f"Portfolio Value: ${total:,.2f}  |  Cash: ${self.cash:,.2f}")
        print(f"Positions ({len(self.positions)}):")
        for t, pos in self.positions.items():
            p   = prices.get(t, pos.avg_cost)
            pnl = (p - pos.avg_cost) * pos.shares
            print(f"  {t:<6}  {pos.shares:>5} shares  "
                  f"cost ${pos.avg_cost:.2f}  now ${p:.2f}  "
                  f"P&L {pnl:+.2f}")
        print(f"{'─'*50}\n")

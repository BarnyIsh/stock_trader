"""
features.py - Technical + fundamental feature engineering for ML model

All features are computed on OHLCV price history and return a
DataFrame with one row per trading day per ticker.

Critical: no future leakage — every feature uses only data available
at or before that day's CLOSE.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend     import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum  import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume    import OnBalanceVolumeIndicator, MFIIndicator
from config       import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD,
    BENCHMARK_TICKER,
)


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download and clean OHLCV from yfinance."""
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    df = df[["open","high","low","close","volume"]].dropna()
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on the OHLCV DataFrame."""
    if len(df) < 50:
        return df

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["ema_9"]  = EMAIndicator(c, window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(c, window=21).ema_indicator()
    df["ema_50"] = EMAIndicator(c, window=50).ema_indicator()
    df["ema_200"]= EMAIndicator(c, window=200).ema_indicator()

    df["sma_20"] = SMAIndicator(c, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(c, window=50).sma_indicator()

    macd = MACD(c, window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    adx = ADXIndicator(h, l, c, window=14)
    df["adx"]    = adx.adx()
    df["adx_pos"]= adx.adx_pos()
    df["adx_neg"]= adx.adx_neg()

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi"]     = RSIIndicator(c, window=RSI_PERIOD).rsi()
    df["rsi_slow"]= RSIIndicator(c, window=21).rsi()

    stoch = StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Rate of change
    df["roc_5"]  = c.pct_change(5)
    df["roc_10"] = c.pct_change(10)
    df["roc_20"] = c.pct_change(20)

    # ── Volatility ────────────────────────────────────────────────────────────
    bb = BollingerBands(c, window=BB_PERIOD, window_dev=BB_STD)
    df["bb_high"]   = bb.bollinger_hband()
    df["bb_low"]    = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_pct"]    = bb.bollinger_pband()      # 0=lower band, 1=upper band
    df["bb_width"]  = bb.bollinger_wband()

    df["atr"]       = AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_pct"]   = df["atr"] / c            # normalised ATR

    df["volatility_20d"] = c.pct_change().rolling(20).std()
    df["volatility_5d"]  = c.pct_change().rolling(5).std()

    # ── Volume ────────────────────────────────────────────────────────────────
    df["obv"]         = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["mfi"]         = MFIIndicator(h, l, c, v, window=14).money_flow_index()
    df["volume_ratio"]= v / v.rolling(20).mean()   # vol surge indicator
    df["volume_sma20"]= v.rolling(20).mean()

    # ── Derived price features ────────────────────────────────────────────────
    df["price_vs_ema50"]  = c / df["ema_50"] - 1
    df["price_vs_ema200"] = c / df["ema_200"] - 1
    df["ema_cross_9_21"]  = (df["ema_9"] > df["ema_21"]).astype(int)
    df["ema_cross_50_200"]= (df["ema_50"] > df["ema_200"]).astype(int)

    # 52-week high/low percentile
    df["pct_from_52w_high"] = c / c.rolling(252).max() - 1
    df["pct_from_52w_low"]  = c / c.rolling(252).min() - 1

    # ── Candlestick patterns (simple) ─────────────────────────────────────────
    df["body"]       = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower_wick"] = df[["close","open"]].min(axis=1) - df["low"]
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)

    return df


def add_labels(
    df: pd.DataFrame,
    forward_days: int = 5,
    min_return: float = 0.02,
) -> pd.DataFrame:
    """
    Add binary label: 1 if forward return ≥ min_return, else 0.
    The label is computed using future data — training only!
    """
    df["future_return"] = df["close"].shift(-forward_days) / df["close"] - 1
    valid = df["future_return"].notna()
    df["label"] = np.nan
    df.loc[valid, "label"] = (
        df.loc[valid, "future_return"] >= min_return
    ).astype(int)
    return df


def build_benchmark_features(
    start: str,
    end: str,
    benchmark: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Build market benchmark features known at each close."""
    bench = download_ohlcv(benchmark, start=start, end=end)
    if bench.empty:
        return pd.DataFrame()

    close = bench["close"]
    out = pd.DataFrame(index=bench.index)
    out["benchmark_ret_5"] = close.pct_change(5)
    out["benchmark_ret_20"] = close.pct_change(20)
    out["benchmark_volatility_20d"] = close.pct_change().rolling(20).std()
    return out


def build_feature_matrix(
    tickers: list[str],
    start: str,
    end: str,
    label: bool = True,
    forward_days: int = 5,
    benchmark: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """
    Build the full feature matrix for a list of tickers over a date range.
    Returns a long-format DataFrame with a 'ticker' column.
    """
    frames = []
    benchmark_df = build_benchmark_features(start, end, benchmark=benchmark)
    for ticker in tickers:
        df = download_ohlcv(ticker, start, end)
        if df.empty or len(df) < 60:
            continue
        df = add_technical_features(df)
        if not benchmark_df.empty:
            df = df.join(benchmark_df, how="left")
            df["relative_roc_5"] = df["roc_5"] - df["benchmark_ret_5"]
            df["relative_roc_20"] = df["roc_20"] - df["benchmark_ret_20"]
        if label:
            df = add_labels(df, forward_days=forward_days)
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames)
    # Drop rows with NaN in feature columns (warm-up period)
    feature_cols = [c for c in out.columns
                    if c not in ("ticker","future_return","label","volume")]
    out = out.dropna(subset=feature_cols)
    if label:
        out = out.dropna(subset=["future_return", "label"])
        out["label"] = out["label"].astype(int)
    return out.reset_index().sort_values(["date", "ticker"]).reset_index(drop=True)


# ─── Feature column selector ─────────────────────────────────────────────────

FEATURE_COLS = [
    "rsi","rsi_slow","stoch_k","stoch_d",
    "macd","macd_signal","macd_hist",
    "adx","adx_pos","adx_neg",
    "bb_pct","bb_width",
    "atr_pct","volatility_20d","volatility_5d",
    "roc_5","roc_10","roc_20",
    "price_vs_ema50","price_vs_ema200",
    "ema_cross_9_21","ema_cross_50_200",
    "pct_from_52w_high","pct_from_52w_low",
    "volume_ratio","obv","mfi",
    "body","upper_wick","lower_wick","is_bullish",
    "benchmark_ret_5","benchmark_ret_20","benchmark_volatility_20d",
    "relative_roc_5","relative_roc_20",
]

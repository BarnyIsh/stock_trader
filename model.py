"""
model.py - Train / evaluate ML model with strict train/test separation

Architecture:
  - Ensemble of RandomForest + GradientBoosting + LogisticRegression
  - Trained ONLY on TRAINING data (2018-2022)
  - Evaluated ONLY on TEST data (2023-present)  ← unseen
  - Walk-forward validation within training window to prevent leakage
  - Feature importances used to explain decisions
"""

import json
import joblib
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration     import CalibratedClassifierCV

from features import (
    build_feature_matrix, build_benchmark_features,
    download_ohlcv, add_technical_features, FEATURE_COLS
)
from config import TRAINING_START, TRAINING_END, TEST_START, TEST_END, BENCHMARK_TICKER

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR = (
    Path("/tmp/stock_trader_logs")
    if os.getenv("VERCEL")
    else Path(__file__).parent / "logs"
)
LOG_DIR.mkdir(exist_ok=True)


# ─── Walk-forward CV (within training window only) ───────────────────────────

def walk_forward_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Time-series cross-validation — no future leakage."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }
    scores = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    model),
        ])
        cv_scores = cross_val_score(
            pipe, X, y, cv=tscv, scoring="roc_auc", n_jobs=-1
        )
        scores[name] = {
            "mean_auc": float(cv_scores.mean()),
            "std_auc":  float(cv_scores.std()),
            "splits":   cv_scores.tolist(),
        }
        print(f"  [{name}] AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return scores


def _new_model_suite() -> dict:
    return {
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1, random_state=42
            )),
        ]),
        "gb": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )),
        ]),
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.5, class_weight="balanced",
                max_iter=500, random_state=42
            )),
        ]),
    }


def _predict_ensemble(models: dict, X: pd.DataFrame) -> np.ndarray:
    p_rf = models["rf"].predict_proba(X)[:, 1]
    p_gb = models["gb"].predict_proba(X)[:, 1]
    p_lr = models["lr"].predict_proba(X)[:, 1]
    return 0.40 * p_rf + 0.40 * p_gb + 0.20 * p_lr


def _signal_profit_pct(df: pd.DataFrame, threshold: float = 0.55) -> float:
    signals = df[df["prob_buy"] >= threshold]
    returns = signals["future_return"].dropna()
    if returns.empty:
        return 0.0
    return float(returns.mean() * 100.0)


def _benchmark_profit_pct(df: pd.DataFrame) -> float:
    if df.empty or "benchmark_ret_5" not in df:
        return 0.0
    daily = (
        df[["date", "benchmark_ret_5"]]
        .dropna()
        .drop_duplicates(subset=["date"])
    )
    if daily.empty:
        return 0.0
    return float(daily["benchmark_ret_5"].mean() * 100.0)


def walk_forward_cv(train_df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Walk-forward validation by date with train/test profit logging."""
    unique_dates = np.array(sorted(pd.to_datetime(train_df["date"]).unique()))
    rows = []
    date_values = pd.to_datetime(train_df["date"])

    for fold, (train_idx, test_idx) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(unique_dates),
        start=1,
    ):
        train_dates = set(unique_dates[train_idx])
        test_dates = set(unique_dates[test_idx])
        fold_train = train_df[date_values.isin(train_dates)]
        fold_test = train_df[date_values.isin(test_dates)]

        models = _new_model_suite()
        X_fold_train = fold_train[FEATURE_COLS].fillna(0)
        y_fold_train = fold_train["label"]
        X_fold_test = fold_test[FEATURE_COLS].fillna(0)
        y_fold_test = fold_test["label"]

        for model in models.values():
            model.fit(X_fold_train, y_fold_train)

        train_prob = _predict_ensemble(models, X_fold_train)
        test_prob = _predict_ensemble(models, X_fold_test)
        train_eval = fold_train.copy()
        test_eval = fold_test.copy()
        train_eval["prob_buy"] = train_prob
        test_eval["prob_buy"] = test_prob

        auc = roc_auc_score(y_fold_test, test_prob)
        row = {
            "fold": fold,
            "train_start": str(fold_train["date"].min().date()),
            "train_end": str(fold_train["date"].max().date()),
            "test_start": str(fold_test["date"].min().date()),
            "test_end": str(fold_test["date"].max().date()),
            "train_samples": int(len(fold_train)),
            "test_samples": int(len(fold_test)),
            "test_auc": float(auc),
            "train_profit_pct": _signal_profit_pct(train_eval),
            "test_profit_pct": _signal_profit_pct(test_eval),
            "benchmark_profit_pct": _benchmark_profit_pct(test_eval),
            "test_signal_count": int((test_prob >= 0.55).sum()),
        }
        rows.append(row)
        print(
            f"  [fold {fold}] AUC={row['test_auc']:.3f}  "
            f"train profit={row['train_profit_pct']:.2f}%  "
            f"test profit={row['test_profit_pct']:.2f}%  "
            f"benchmark={row['benchmark_profit_pct']:.2f}%"
        )

    out = pd.DataFrame(rows)
    return {
        "mean_auc": float(out["test_auc"].mean()),
        "mean_train_profit_pct": float(out["train_profit_pct"].mean()),
        "mean_test_profit_pct": float(out["test_profit_pct"].mean()),
        "mean_benchmark_profit_pct": float(out["benchmark_profit_pct"].mean()),
        "folds": rows,
    }


def _calibrated_feature_importances(calibrated: CalibratedClassifierCV) -> np.ndarray:
    """Average feature importances from fitted calibrated base estimators."""
    fitted = getattr(calibrated, "calibrated_classifiers_", [])
    importances = []
    for item in fitted:
        estimator = getattr(item, "estimator", None)
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            importances.append(estimator.feature_importances_)
    if importances:
        return np.mean(importances, axis=0)

    estimator = getattr(calibrated, "estimator", None)
    if estimator is not None and hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_

    return np.zeros(len(FEATURE_COLS))


# ─── Full model train ─────────────────────────────────────────────────────────

def train_model(
    tickers: list[str],
    forward_days: int = 5,
    min_return: float = 0.02,
    verbose: bool = True,
) -> dict:
    """
    1. Build feature matrix on TRAINING data only.
    2. Walk-forward CV to select best hyper-params.
    3. Retrain full ensemble on all training data.
    4. Persist model + metadata.
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"TRAINING  ({TRAINING_START} → {TRAINING_END})")
        print("=" * 60)

    # Build training features
    train_df = build_feature_matrix(
        tickers, TRAINING_START, TRAINING_END,
        label=True, forward_days=forward_days
    )
    if train_df.empty:
        raise ValueError("No training data found for provided tickers.")

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df["label"]

    if verbose:
        print(f"Training samples: {len(X_train):,}  |  "
              f"Positive rate: {y_train.mean():.1%}")

    # Walk-forward CV (training window only)
    if verbose:
        print("\nWalk-forward CV results:")
    cv_scores = walk_forward_cv(train_df)

    # ── Ensemble definition ──────────────────────────────────────────────────
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=15,
                class_weight="balanced", n_jobs=-1, random_state=42
            ), cv=3
        )),
    ])
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            ), cv=3
        )),
    ])
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=0.5, class_weight="balanced",
            max_iter=500, random_state=42
        )),
    ])

    if verbose:
        print("\nFitting ensemble on full training set…")
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # Feature importances from tree models
    rf_fi = _calibrated_feature_importances(rf.named_steps["clf"])
    gb_fi = _calibrated_feature_importances(gb.named_steps["clf"])
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "rf_importance": rf_fi,
        "gb_importance": gb_fi,
        "mean_importance": (rf_fi + gb_fi) / 2,
    }).sort_values("mean_importance", ascending=False)

    if verbose:
        print("\nTop 10 features by importance:")
        print(fi_df.head(10)[["feature","mean_importance"]].to_string(index=False))

    # Persist models
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    joblib.dump(rf, MODEL_DIR / f"rf_{ts}.pkl")
    joblib.dump(gb, MODEL_DIR / f"gb_{ts}.pkl")
    joblib.dump(lr, MODEL_DIR / f"lr_{ts}.pkl")

    # Save latest symlinks
    joblib.dump(rf, MODEL_DIR / "rf_latest.pkl")
    joblib.dump(gb, MODEL_DIR / "gb_latest.pkl")
    joblib.dump(lr, MODEL_DIR / "lr_latest.pkl")

    metadata = {
        "trained_at":    ts,
        "train_start":   TRAINING_START,
        "train_end":     TRAINING_END,
        "tickers":       tickers,
        "n_samples":     int(len(X_train)),
        "forward_days":  forward_days,
        "min_return":    min_return,
        "benchmark":     BENCHMARK_TICKER,
        "cv_scores":     cv_scores,
        "feature_cols":  FEATURE_COLS,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    log_path = LOG_DIR / f"training_{ts}.json"
    log_path.write_text(json.dumps(metadata, indent=2, default=str))
    pd.DataFrame(cv_scores["folds"]).to_csv(
        LOG_DIR / f"training_folds_{ts}.csv", index=False
    )
    if verbose:
        print(f"\nModels saved to {MODEL_DIR}/")
        print(f"Training logs saved to {log_path}")

    return {"models": {"rf": rf, "gb": gb, "lr": lr}, "metadata": metadata, "fi": fi_df}


# ─── Test evaluation (UNSEEN data) ───────────────────────────────────────────

def evaluate_on_test(
    model_bundle: dict,
    tickers: list[str],
    forward_days: int = 5,
    min_return: float = 0.02,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate the trained ensemble on strictly unseen TEST data.
    Returns per-day signal DataFrame.
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"TESTING ON UNSEEN DATA ({TEST_START} → {TEST_END})")
        print("=" * 60)

    models  = model_bundle["models"]
    feature_cols = model_bundle["metadata"]["feature_cols"]

    test_df = build_feature_matrix(
        tickers, TEST_START, TEST_END,
        label=True, forward_days=forward_days
    )
    if test_df.empty:
        print("[warn] No test data available yet.")
        return pd.DataFrame()

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["label"]

    # Ensemble predict
    p_rf = models["rf"].predict_proba(X_test)[:, 1]
    p_gb = models["gb"].predict_proba(X_test)[:, 1]
    p_lr = models["lr"].predict_proba(X_test)[:, 1]
    ensemble_prob = _predict_ensemble(models, X_test)

    test_df["prob_buy"]    = ensemble_prob
    test_df["signal"]      = (ensemble_prob >= 0.55).astype(int)
    test_df["prob_rf"]     = p_rf
    test_df["prob_gb"]     = p_gb
    test_df["prob_lr"]     = p_lr

    if verbose and len(y_test) > 0:
        auc = roc_auc_score(y_test, ensemble_prob)
        preds = (ensemble_prob >= 0.55).astype(int)
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
        print(f"F1:        {f1_score(y_test, preds, zero_division=0):.4f}")
        print(f"Signal profit: {_signal_profit_pct(test_df):.2f}%")
        print(f"{BENCHMARK_TICKER} benchmark: {_benchmark_profit_pct(test_df):.2f}%")
        print(f"\nClassification Report:")
        print(classification_report(y_test, preds, target_names=["Hold","Buy"]))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_log = {
            "evaluated_at": ts,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "tickers": tickers,
            "signal_profit_pct": _signal_profit_pct(test_df),
            "benchmark_profit_pct": _benchmark_profit_pct(test_df),
            "signal_count": int(test_df["signal"].sum()),
            "rows": int(len(test_df)),
            "roc_auc": float(auc),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }
        (LOG_DIR / f"test_evaluation_{ts}.json").write_text(
            json.dumps(eval_log, indent=2, default=str)
        )

    return test_df


# ─── Live scoring (today, no label) ──────────────────────────────────────────

def score_today(
    tickers: list[str],
    model_bundle: dict | None = None,
) -> pd.DataFrame:
    """
    Score a list of tickers for TODAY's trading session.
    Uses batch download for efficiency with large universes.
    Returns DataFrame ranked by buy probability.
    """
    if model_bundle is None:
        try:
            rf = joblib.load(MODEL_DIR / "rf_latest.pkl")
            gb = joblib.load(MODEL_DIR / "gb_latest.pkl")
            lr = joblib.load(MODEL_DIR / "lr_latest.pkl")
            meta = json.loads((MODEL_DIR / "metadata.json").read_text())
            feature_cols = meta["feature_cols"]
        except FileNotFoundError:
            raise RuntimeError("No trained model found. Run train_model() first.")
    else:
        rf = model_bundle["models"]["rf"]
        gb = model_bundle["models"]["gb"]
        lr = model_bundle["models"]["lr"]
        feature_cols = model_bundle["metadata"]["feature_cols"]

    import yfinance as yf

    # Use only last 10 months of data for scoring (enough for all indicators)
    from datetime import datetime, timedelta
    score_start = (datetime.today() - timedelta(days=300)).strftime("%Y-%m-%d")
    score_end = datetime.today().strftime("%Y-%m-%d")

    # Batch download in chunks to avoid timeouts on serverless
    chunk_size = 40
    all_frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(
                chunk, start=score_start, end=score_end,
                auto_adjust=True, progress=False, group_by="ticker", threads=True,
            )
            if not data.empty:
                all_frames.append((chunk, data))
        except Exception as exc:
            print(f"  [warn] Chunk {i}-{i+len(chunk)} download failed: {exc}")

    if not all_frames:
        print("  [error] No price data downloaded.")
        return pd.DataFrame()

    benchmark_df = build_benchmark_features(score_start, score_end)

    rows = []
    for chunk_tickers, batch_data in all_frames:
        for ticker in chunk_tickers:
            try:
                # Extract single ticker from batch
                if isinstance(batch_data.columns, pd.MultiIndex):
                    if ticker not in batch_data.columns.get_level_values(0):
                        continue
                    df = batch_data[ticker].copy()
                else:
                    # Non-multi-index (shouldn't happen with group_by='ticker')
                    df = batch_data.copy()

                if df.empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                if len(df) < 60:
                    continue

                df = add_technical_features(df)
                if not benchmark_df.empty:
                    df = df.join(benchmark_df, how="left")
                    df["relative_roc_5"] = df["roc_5"] - df["benchmark_ret_5"]
                    df["relative_roc_20"] = df["roc_20"] - df["benchmark_ret_20"]

                # Handle feature cols that may not exist in older models
                available_features = [c for c in feature_cols if c in df.columns]
                if len(available_features) < len(feature_cols) * 0.8:
                    continue

                # Drop rows where core features are NaN (warm-up period)
                # but don't require ALL features — some (52-week) need more history
                core_features = [c for c in available_features
                                 if not c.startswith("pct_from_52w")]
                df = df.dropna(subset=core_features)
                if df.empty:
                    continue

                last = df.iloc[[-1]][available_features].fillna(0)
                # Pad missing features with 0
                for col in feature_cols:
                    if col not in last.columns:
                        last[col] = 0.0
                last = last[feature_cols]

                p_rf = rf.predict_proba(last)[0, 1]
                p_gb = gb.predict_proba(last)[0, 1]
                p_lr = lr.predict_proba(last)[0, 1]
                prob = 0.40 * p_rf + 0.40 * p_gb + 0.20 * p_lr
                rows.append({
                    "ticker":    ticker,
                    "date":      df.index[-1],
                    "price":     float(df["close"].iloc[-1]),
                    "prob_buy":  float(prob),
                    "prob_rf":   float(p_rf),
                    "prob_gb":   float(p_gb),
                    "prob_lr":   float(p_lr),
                    "rsi":       float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0,
                    "bb_pct":    float(df["bb_pct"].iloc[-1]) if "bb_pct" in df.columns else 0.5,
                    "macd_hist": float(df["macd_hist"].iloc[-1]) if "macd_hist" in df.columns else 0.0,
                    "roc_5":     float(df["roc_5"].iloc[-1]) if "roc_5" in df.columns else 0.0,
                })
            except Exception:
                pass  # skip silently for large universes

    if not rows:
        return pd.DataFrame()

    print(f"  Scored {len(rows)} tickers successfully.")
    return pd.DataFrame(rows).sort_values("prob_buy", ascending=False).reset_index(drop=True)

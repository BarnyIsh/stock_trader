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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration     import CalibratedClassifierCV

from features import (
    build_feature_matrix, download_ohlcv, add_technical_features, FEATURE_COLS
)
from config import TRAINING_START, TRAINING_END, TEST_START, TEST_END

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


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
    cv_scores = walk_forward_cv(X_train, y_train)

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
    rf_fi = rf.named_steps["clf"].estimator.feature_importances_
    gb_fi = gb.named_steps["clf"].estimator.feature_importances_
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
        "cv_scores":     cv_scores,
        "feature_cols":  FEATURE_COLS,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    if verbose:
        print(f"\nModels saved to {MODEL_DIR}/")

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
    ensemble_prob = (0.40 * p_rf + 0.40 * p_gb + 0.20 * p_lr)

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
        print(f"\nClassification Report:")
        print(classification_report(y_test, preds, target_names=["Hold","Buy"]))

    return test_df


# ─── Live scoring (today, no label) ──────────────────────────────────────────

def score_today(
    tickers: list[str],
    model_bundle: dict | None = None,
) -> pd.DataFrame:
    """
    Score a list of tickers for TODAY's trading session.
    Returns DataFrame ranked by buy probability.
    """
    if model_bundle is None:
        # Load latest persisted models
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

    rows = []
    for ticker in tickers:
        try:
            df = download_ohlcv(ticker, start=TEST_START, end=TEST_END)
            if len(df) < 60:
                continue
            df = add_technical_features(df)
            df = df.dropna(subset=feature_cols)
            if df.empty:
                continue
            last = df.iloc[[-1]][feature_cols].fillna(0)
            p_rf = rf.predict_proba(last)[0, 1]
            p_gb = gb.predict_proba(last)[0, 1]
            p_lr = lr.predict_proba(last)[0, 1]
            prob  = 0.40 * p_rf + 0.40 * p_gb + 0.20 * p_lr
            rows.append({
                "ticker":    ticker,
                "date":      df.index[-1],
                "price":     float(df["close"].iloc[-1]),
                "prob_buy":  float(prob),
                "prob_rf":   float(p_rf),
                "prob_gb":   float(p_gb),
                "prob_lr":   float(p_lr),
                "rsi":       float(df["rsi"].iloc[-1]),
                "bb_pct":    float(df["bb_pct"].iloc[-1]),
                "macd_hist": float(df["macd_hist"].iloc[-1]),
                "roc_5":     float(df["roc_5"].iloc[-1]),
            })
        except Exception as e:
            print(f"  [skip] {ticker}: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("prob_buy", ascending=False).reset_index(drop=True)

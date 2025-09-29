"""Machine learning models for ETF forecasting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "volatility",
    "max_drawdown",
    "sharpe",
    "sortino",
    "vwap_gap",
    "volume_z",
    "obv_trend",
    "pattern_confidence",
    "sentiment_score",
]
TARGET_COLUMN = "forward_return"


def build_model() -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )
    gbr = GradientBoostingRegressor(random_state=42, n_estimators=500, learning_rate=0.03, max_depth=3)
    pipe = Pipeline([
        ("transform", transformer),
        ("model", gbr),
    ])
    return pipe


def train_model(df: pd.DataFrame, model_path: str) -> Tuple[Pipeline, float]:
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if len(df) < 200:
        raise ValueError("Not enough observations to train ML model")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    score = r2_score(y_valid, preds)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    logger.info("Trained ML model with R2 %.3f saved to %s", score, model_path)
    return model, score


def load_or_train_model(df: pd.DataFrame, model_path: str) -> Tuple[Pipeline, float]:
    path = Path(model_path)
    if path.exists():
        try:
            model = load(path)
            logger.info("Loaded ML model from %s", model_path)
            return model, float("nan")
        except Exception:
            logger.warning("Existing ML model corrupted. Retraining...")
    return train_model(df, model_path)


def predict_returns(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    missing = [col for col in FEATURE_COLUMNS if col not in features.columns]
    for col in missing:
        features[col] = 0.0
    preds = model.predict(features[FEATURE_COLUMNS].fillna(0))
    return preds

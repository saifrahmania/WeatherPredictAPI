# app/main.py
from fastapi import FastAPI, Query, HTTPException
from datetime import date as date_type
from typing import Dict, Any
import logging
import math
from pathlib import Path

import pandas as pd

from .schemas import (
    HealthResponse,
    RainResponse, RainPrediction,
    PrecipResponse, PrecipPrediction,
)
from .model_loader import load_models
from .feature_builder import FeatureBuilder

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("weather-api")

# ---------------- App ----------------
app = FastAPI(
    title="Open Meteo – ML as a Service",
    description=(
        "Two models for Sydney:\n"
        "1) Rain-or-not in exactly +7 days (classification)\n"
        "2) 3-day cumulative precipitation (mm) for next 3 days (regression)\n\n"
        "Endpoints:\n"
        " - GET /               -> About & usage\n"
        " - GET /health/        -> 200 OK\n"
        " - GET /predict/rain/  -> ?date=YYYY-MM-DD\n"
        " - GET /predict/precipitation/fall/ -> ?date=YYYY-MM-DD\n"
    ),
    version="1.0.0",
)

# ---------------- Load models once ----------------
RAIN_MODEL, PREC_MODEL, RAIN_THRESHOLD = load_models()

# ---------------- Dataset coverage (min/max supported date) ----------------
RAIN_FEATS = Path("data/processed/features_rain_daily.parquet")
PREC_FEATS = Path("data/processed/features_precip_daily.parquet")

def _date_range_from_parquet(p: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not p.exists():
        raise RuntimeError(f"Required feature file missing: {p}")
    df = pd.read_parquet(p, columns=None)
    for c in df.columns:
        cl = str(c).lower()
        if cl in {"date", "date_daily"} or pd.api.types.is_datetime64_any_dtype(df[c]):
            d = pd.to_datetime(df[c]).dt.tz_localize(None).dt.normalize()
            return d.min(), d.max()
    raise RuntimeError(f"No date-like column in {p}")

try:
    rmin, rmax = _date_range_from_parquet(RAIN_FEATS)
    pmin, pmax = _date_range_from_parquet(PREC_FEATS)
    DATA_MIN = min(rmin, pmin)
    DATA_MAX = max(rmax, pmax)
    log.info("Feature coverage: %s → %s", DATA_MIN.date(), DATA_MAX.date())
except Exception as e:
    log.error("Failed to determine feature date range: %s", e)
    raise

# ---------------- Helpers ----------------
def _parse_date_str(d: str) -> pd.Timestamp:
    """Parse 'YYYY-MM-DD' into a normalized Timestamp (HTTP 400 if invalid)."""
    try:
        ts = pd.to_datetime(d, format="%Y-%m-%d", errors="raise")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Expected 'YYYY-MM-DD'.")
    return ts.tz_localize(None).normalize()

def _validate_not_before_min(d0: pd.Timestamp):
    """
    Only enforce lower bound. We allow dates AFTER DATA_MAX; the builder will
    use the latest available feature row ≤ the requested date.
    """
    if d0 < DATA_MIN:
        raise HTTPException(
            status_code=422,
            detail=f"Date {d0.date()} is before supported range start {DATA_MIN.date()}."
        )

def _no_future_leakage(feature_end: pd.Timestamp, target_start: pd.Timestamp):
    if feature_end >= target_start:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Feature end {feature_end.date()} must be strictly "
                f"before target start {target_start.date()}."
            ),
        )

# ---------------- Routes ----------------
@app.get("/", response_model=Dict[str, Any])
def home() -> Dict[str, Any]:
    example_rain = {
        "input_date": "2025-01-01",
        "prediction": {"date": "2025-01-08", "will_rain": True},
    }
    example_prec = {
        "input_date": "2025-01-01",
        "prediction": {
            "start_date": "2025-01-02",
            "end_date": "2025-01-04",
            "precipitation_fall": 28.2,
        },
    }
    return {
        "project": "Open Meteo – Weather Predictions API (Sydney)",
        "github": "YOUR_API_REPO_LINK_HERE",
        "endpoints": {
            "/": "this message",
            "/health/": "GET – service status",
            "/predict/rain/": "GET – ?date=YYYY-MM-DD (rain-or-not for date+7)",
            "/predict/precipitation/fall/": "GET – ?date=YYYY-MM-DD (3-day cumulative precipitation)",
        },
        "notes": (
            "Dates after data_max are accepted; the latest available feature row "
            f"(≤ requested date) is used. Dates before {DATA_MIN.date()} are rejected."
        ),
        "coverage": {
            "data_min": str(DATA_MIN.date()),
            "data_max": str(DATA_MAX.date()),
        },
        "examples": {"rain": example_rain, "precipitation": example_prec},
    }

@app.get("/health/", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.get("/predict/rain/", response_model=RainResponse)
def predict_rain(date: str = Query(..., description="YYYY-MM-DD")) -> RainResponse:
    d0 = _parse_date_str(date)
    _validate_not_before_min(d0)

    fb = FeatureBuilder("rain").build_for_date(date)
    target_date = d0 + pd.Timedelta(days=7)
    _no_future_leakage(fb.feature_end_date, target_date)

    if hasattr(RAIN_MODEL, "predict_proba"):
        proba = float(RAIN_MODEL.predict_proba(fb.X)[:, 1][0])
    elif hasattr(RAIN_MODEL, "decision_function"):
        score = float(RAIN_MODEL.decision_function(fb.X)[0])
        proba = 1.0 / (1.0 + math.exp(-score))
    else:
        proba = float(RAIN_MODEL.predict(fb.X)[0])

    will_rain = bool(proba >= RAIN_THRESHOLD)
    log.info("RAIN request date=%s -> prob=%.4f will_rain=%s", d0.date(), proba, will_rain)

    return RainResponse(
        input_date=d0.date(),
        prediction=RainPrediction(date=target_date.date(), will_rain=will_rain),
    )

@app.get("/predict/precipitation/fall/", response_model=PrecipResponse)
def predict_precipitation(date: str = Query(..., description="YYYY-MM-DD")) -> PrecipResponse:
    d0 = _parse_date_str(date)
    _validate_not_before_min(d0)

    fb = FeatureBuilder("precip").build_for_date(date)
    start_date = d0 + pd.Timedelta(days=1)
    end_date   = d0 + pd.Timedelta(days=3)
    _no_future_leakage(fb.feature_end_date, start_date)

    yhat = float(PREC_MODEL.predict(fb.X)[0])
    yhat = max(0.0, yhat)

    log.info("PRECIP request date=%s -> yhat=%.3f", d0.date(), yhat)

    return PrecipResponse(
        input_date=d0.date(),
        prediction=PrecipPrediction(
            start_date=start_date.date(),
            end_date=end_date.date(),
            precipitation_fall=round(yhat, 2),
        ),
    )

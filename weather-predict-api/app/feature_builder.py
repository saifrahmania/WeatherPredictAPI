# app/feature_builder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# ---------------------------
# Leak/label guards + aligner
# ---------------------------

LEAKY_TARGET_COLS = {
    "date", "will_rain", "will_rain_plus7", "y_rain_binary",
    "precip_3d", "y_precip", "precip_d1", "precip_d2", "precip_d3",
    "rain_in_7d", "precip_plus7", "target", "label",
}

def _expected_columns_from_pipeline(model) -> list[str] | None:
    """
    Try to extract the exact feature list the trained pipeline expects.
    Works with model.feature_names_in_ (when present) or by reading a
    ColumnTransformer inside an sklearn Pipeline.
    """
    # sklearn >= 1.0 often sets this attribute
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # Otherwise try to find a ColumnTransformer in a Pipeline
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        if isinstance(model, Pipeline) and hasattr(model, "named_steps"):
            for _, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    selected: list[str] = []
                    for _, cols_sel, _ in step.transformers_:
                        if cols_sel == "drop":
                            continue
                        if isinstance(cols_sel, (list, tuple)):
                            selected.extend(list(cols_sel))
                    if selected:
                        # preserve order & uniqueness
                        return list(dict.fromkeys(selected))
    except Exception:
        pass

    return None

def align_to_model_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align X to the exact columns the model expects.

    - If we can read the expected list, reindex to it and add missing cols as 0.0.
    - If we cannot, just drop obvious leaky columns (best-effort).
    """
    expected = _expected_columns_from_pipeline(model)

    # Always drop known leak/label columns unless they are explicitly expected
    leaks_to_drop = [
        c for c in X.columns
        if (c in LEAKY_TARGET_COLS) and (expected is None or c not in expected)
    ]
    X = X.drop(columns=leaks_to_drop, errors="ignore")

    if expected is None:
        return X  # no strict alignment possible, but at least we removed leaks

    # Add missing expected columns with 0.0
    for c in expected:
        if c not in X.columns:
            X[c] = 0.0

    # Reindex strictly to the expected order (ColumnTransformer requires exact match)
    return X.reindex(columns=expected)


# ---------------------------
# Local config for features
# ---------------------------

PROCESSED_DIR = Path("data/processed")
FEATURES_RAIN_PARQUET = "features_rain_daily.parquet"
FEATURES_PRECIP_PARQUET = "features_precip_daily.parquet"


# ---------------------------
# Utilities
# ---------------------------

def _parse_date(d: str) -> pd.Timestamp:
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date string: {d}. Expected YYYY-MM-DD.")
    return ts.normalize()

def _pick_date_column(df: pd.DataFrame) -> str:
    if "date_daily" in df.columns:
        return "date_daily"
    if "date" in df.columns:
        return "date"
    # auto-detect any datetime-like column
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    raise ValueError("No date-like column found in features dataframe.")


# ---------------------------
# Feature bundle
# ---------------------------

@dataclass
class FeatureBundle:
    X: pd.DataFrame
    feature_names: list
    feature_end_date: pd.Timestamp


# ---------------------------
# FeatureBuilder
# ---------------------------

class FeatureBuilder:
    """
    Builds one-row feature frames for a given anchor date (no future leakage).
    - task='rain'   -> for the +7d classification
    - task='precip' -> for the 3-day cumulative precipitation regression
    """
    def __init__(self, task: str):
        assert task in {"rain", "precip"}, "task must be 'rain' or 'precip'"
        self.task = task

    def _load_parquet_row_for_date(self, fpath: Path, anchor: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Timestamp]:
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing features file: {fpath}. "
                "Create it or ensure your repo contains data/processed/*.parquet."
            )
        df = pd.read_parquet(fpath)
        dcol = _pick_date_column(df)
        df[dcol] = pd.to_datetime(df[dcol]).dt.normalize()

        # Exact match; otherwise take the last row <= requested anchor date
        row = df.loc[df[dcol] == anchor]
        if row.empty:
            row = df.loc[df[dcol] <= anchor].sort_values(dcol).tail(1)
        if row.empty:
            raise ValueError(f"No feature row available for {anchor.date()} in {fpath} (using column '{dcol}').")

        feat_end = pd.to_datetime(row[dcol].iloc[0]).normalize()

        # Drop date + common label/leak columns
        drop_cols = {dcol} | LEAKY_TARGET_COLS
        X = row.drop(columns=[c for c in row.columns if c in drop_cols], errors="ignore").reset_index(drop=True)
        return X, feat_end

    def build_for_date(self, input_date: str) -> FeatureBundle:
        anchor = _parse_date(input_date)

        if self.task == "rain":
            fpath = PROCESSED_DIR / FEATURES_RAIN_PARQUET
            X, feat_end = self._load_parquet_row_for_date(fpath, anchor)

            # Align to the trained classification pipeline (important)
            try:
                # Local import to avoid circulars at module import time
                from .model_loader import load_models
                _RAIN_MODEL, _, _ = load_models()
                X = align_to_model_features(X.copy(), _RAIN_MODEL)
            except Exception:
                # If anything fails, still return X (we already dropped leaks)
                pass

            return FeatureBundle(X=X, feature_names=list(X.columns), feature_end_date=feat_end)

        else:  # self.task == "precip"
            fpath = PROCESSED_DIR / FEATURES_PRECIP_PARQUET
            X, feat_end = self._load_parquet_row_for_date(fpath, anchor)

            # Keep it clean for regression too
            X = X.drop(columns=[c for c in X.columns if c in LEAKY_TARGET_COLS], errors="ignore")

            try:
                from .model_loader import load_models
                _, _PREC_MODEL, _ = load_models()
                X = align_to_model_features(X.copy(), _PREC_MODEL)
            except Exception:
                pass

            return FeatureBundle(X=X, feature_names=list(X.columns), feature_end_date=feat_end)
from pathlib import Path
from typing import Optional, Tuple
import json
from joblib import load

ROOT = Path(__file__).resolve().parent.parent

RAIN_DIR   = ROOT / "models" / "rain_or_not"
PREC_DIR   = ROOT / "models" / "precipitation_fall"

RAIN_MODEL_PATH  = RAIN_DIR / "best_model.joblib"
PREC_MODEL_PATH  = PREC_DIR / "best_model.joblib"
RAIN_THRESH_PATH = RAIN_DIR / "threshold.json"

def load_models() -> Tuple[object, object, float]:
    if not RAIN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Rain model not found: {RAIN_MODEL_PATH}")
    if not PREC_MODEL_PATH.exists():
        raise FileNotFoundError(f"Precip model not found: {PREC_MODEL_PATH}")

    rain_model = load(RAIN_MODEL_PATH)
    precip_model = load(PREC_MODEL_PATH)

    # optional threshold
    threshold = 0.5
    if RAIN_THRESH_PATH.exists():
        try:
            data = json.loads(RAIN_THRESH_PATH.read_text())
            # support {"threshold": x} or {"rain":{"threshold":x}}
            threshold = data.get("threshold", data.get("rain", {}).get("threshold", 0.5))
        except Exception:
            pass

    return rain_model, precip_model, float(threshold)

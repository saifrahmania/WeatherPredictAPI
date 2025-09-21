from fastapi import FastAPI, Query, HTTPException
from datetime import date
from app.schemas import HealthResponse, RainResponse, PrecipResponse
from app.model_loader import ModelStore
from app.utils import plus_days

app = FastAPI(
    title="Open Meteo — Rain & Precipitation API",
    description="Predicts (1) rain in +7 days and (2) 3-day precipitation total for Sydney.",
    version="1.0.0",
)

@app.on_event("startup")
def _startup():
    ModelStore.load()

@app.get("/", tags=["info"])
def root():
    return {
        "project": "Open Meteo – ML as a Service",
        "endpoints": {
            "/health/": {"method": "GET"},
            "/predict/rain/": {"method": "GET", "params": {"date": "YYYY-MM-DD"}},
            "/predict/precipitation/fall": {"method": "GET", "params": {"date": "YYYY-MM-DD"}},
        },
        "output_examples": {
            "/predict/rain/": {
                "input_date": "2023-01-01",
                "prediction": {"date": "2023-01-08", "will_rain": True}
            },
            "/predict/precipitation/fall": {
                "input_date": "2023-01-01",
                "prediction": {"start_date": "2023-01-02", "end_date_date": "2023-01-04", "precipitation_fall": 28.2}
            }
        },
        "github": "SEE github.txt"
    }

@app.get("/health/", response_model=HealthResponse, tags=["info"])
def health():
    return HealthResponse(status="ok")

@app.get("/predict/rain/", response_model=RainResponse, tags=["predict"])
def predict_rain(date: date = Query(..., description="YYYY-MM-DD")):
    clf, _ = ModelStore.load()
    try:
        # TODO: replace with real features for the given date
        y = clf.predict([[date.toordinal()]])[0]
        will_rain = bool(int(y))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")
    return {"input_date": date, "prediction": {"date": plus_days(date, 7), "will_rain": will_rain}}

@app.get("/predict/precipitation/fall", response_model=PrecipResponse, tags=["predict"])
def predict_precip(date: date = Query(..., description="YYYY-MM-DD")):
    _, reg = ModelStore.load()
    try:
        # TODO: replace with real features for the given date
        yhat = float(reg.predict([[date.toordinal()]])[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")
    return {
        "input_date": date,
        "prediction": {
            "start_date": plus_days(date, 1),
            "end_date_date": plus_days(date, 3),
            "precipitation_fall": round(yhat, 2)
        }
    }

# app/schemas.py
from pydantic import BaseModel
from datetime import date
from typing import Literal

class HealthResponse(BaseModel):
    status: Literal["ok"]

# ---- Rain (classification) ----
class RainPrediction(BaseModel):
    date: date            # input_date + 7
    will_rain: bool

class RainResponse(BaseModel):
    input_date: date
    prediction: RainPrediction

# ---- Precipitation (regression) ----
class PrecipPrediction(BaseModel):
    start_date: date      # input_date + 1
    end_date: date        # input_date + 3
    precipitation_fall: float

class PrecipResponse(BaseModel):
    input_date: date
    prediction: PrecipPrediction

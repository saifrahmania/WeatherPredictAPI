from pydantic import BaseModel, Field
from datetime import date

class HealthResponse(BaseModel):
    status: str

class RainResponse(BaseModel):
    input_date: date
    prediction: dict

class PrecipResponse(BaseModel):
    input_date: date
    prediction: dict

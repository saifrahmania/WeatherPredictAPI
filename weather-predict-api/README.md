# WeatherPredictAPI

# Create venv
python -m venv .venv_api
.\.venv_api\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000

# Install deps
pip install -r requirements.txt

# Run API
python -m uvicorn app.main:app --reload --port 8000

# Test Endpoints

Once the server is running and you see:

You can test the following endpoints in your browser or with `curl`:


- **Health check**  
  [http://127.0.0.1:8000/health/](http://127.0.0.1:8000/health/)  
  Returns service status (`{"status": "ok"}`).

- **Rain prediction (+7 days)**  
  [http://127.0.0.1:8000/predict/rain/?date=2025-01-03](http://127.0.0.1:8000/predict/rain/?date=2025-01-03)  
  Input a date (YYYY-MM-DD). Returns whether it will rain exactly 7 days later.

- **3-day precipitation total**  
  [http://127.0.0.1:8000/predict/precipitation/fall/?date=2025-01-03](http://127.0.0.1:8000/predict/precipitation/fall/?date=2025-01-03)  
  Input a date (YYYY-MM-DD). Returns cumulative rainfall (mm) for the next 3 days.

- **API Documentation**  
  [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 
  Shows interactive API docs (Swagger UI).
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_root_ok():
    assert client.get("/").status_code == 200

def test_predict_rain_bad_date():
    r = client.get("/predict/rain/", params={"date": "bad"})
    assert r.status_code in (400, 422)

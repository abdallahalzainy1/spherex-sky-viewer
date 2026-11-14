from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)


def test_demo_frames_and_modes():
    for mode in ("single", "diff"):
        r = c.get("/api/frames", params={"target": "133.79 -7.25", "size": 3, "mode": mode, "demo": True})
        assert r.status_code == 200 and r.json()["n"] == 14


def test_binning_and_detector():
    r = c.get("/api/frames", params={"target": "10 20", "demo": True, "bin": 2})
    assert r.json()["n"] == 7
    r = c.get("/api/frames", params={"target": "10 20", "demo": True, "det": 1})
    assert 0 < r.json()["n"] < 14


def test_rejects_bad_input():
    assert c.get("/api/frames", params={"target": "10 20", "size": 500, "demo": True}).status_code == 422

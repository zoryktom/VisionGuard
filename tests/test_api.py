"""FastAPI tests."""

from __future__ import annotations

import io

import pytest
from api.deps import reset_pipeline
from api.main import app
from fastapi.testclient import TestClient
from PIL import Image

from visionguard.capture import synthetic_frame


@pytest.fixture()
def client() -> TestClient:
    reset_pipeline()
    with TestClient(app) as test_client:
        yield test_client
    reset_pipeline()


def test_health(client: TestClient) -> None:
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_metrics_prometheus(client: TestClient) -> None:
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    assert "visionguard_fps" in res.text


def test_infer_image(client: TestClient) -> None:
    frame = synthetic_frame(320, 192, seed=3)
    image = Image.fromarray(frame[:, :, ::-1])
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    res = client.post("/api/v1/infer/image", files={"file": ("frame.jpg", buf, "image/jpeg")})
    assert res.status_code == 200
    body = res.json()
    assert "risk_score" in body
    assert "detections" in body


def test_dashboard_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "VISIONGUARD" in res.text


def test_snapshot(client: TestClient) -> None:
    res = client.get("/api/v1/stream/snapshot")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/jpeg")

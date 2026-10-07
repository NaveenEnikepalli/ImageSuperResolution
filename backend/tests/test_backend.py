"""Unit and Integration Tests for FastAPI Backend.

Author: Antigravity
Purpose: Validate FastAPI endpoints (/health, /api/health, /api/models, /enhance, /api/enhance),
status codes, scale validation, custom response headers, and output non-corruption regression assertions.
"""

import sys
import io
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert 2 in data["supported_scales"] or "2x" in data["supported_scales"]


def test_health_endpoints():
    """Test health check GET /health and GET /api/health endpoints."""
    for path in ["/health", "/api/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "ok"]
        assert 2 in data["supported_scales"]


def test_models_endpoint():
    """Test GET /api/models endpoint."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert set(data["supported_scales"]) == {2, 4, 8}
    assert "2" in data["models"] or 2 in data["models"]


def test_enhance_endpoint_invalid_scale():
    """Test POST /api/enhance with an invalid scale parameter (e.g. scale=3)."""
    img = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/enhance",
        files={"image": ("test.png", buf, "image/png")},
        data={"scale": 3},
    )
    assert response.status_code == 400
    assert "Unsupported scale factor" in response.json()["detail"]


def test_enhance_endpoint_scale_2_regression():
    """Regression test for Student x2: valid PNG, RGB mode, correct dimensions, and non-corrupt pixels."""
    img = Image.new("RGB", (32, 32), color=(180, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/enhance",
        files={"image": ("test.png", buf, "image/png")},
        data={"scale": 2},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "X-Inference-Time" in response.headers
    assert response.headers["X-Scale-Factor"] == "2"
    assert response.headers["X-Model-Used"] == "Student x2"

    out_bytes = response.content
    out_img = Image.open(io.BytesIO(out_bytes))
    out_img.verify()

    out_img = Image.open(io.BytesIO(out_bytes))
    assert out_img.mode == "RGB"
    assert out_img.size == (64, 64)

    out_np = np.array(out_img)
    assert out_np.shape == (64, 64, 3)
    assert out_np.min() >= 0 and out_np.max() <= 255
    # Assert non-corrupt output: image pixel mean should reflect input intensity, not dark/black garbage
    assert out_np.mean() > 30.0


def test_enhance_endpoint_scale_4_regression():
    """Regression test for Student x4: valid PNG, RGB mode, correct dimensions, and non-corrupt pixels."""
    img = Image.new("RGB", (16, 16), color=(50, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/enhance",
        files={"file": ("test.png", buf, "image/png")},
        data={"scale": 4},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["X-Scale-Factor"] == "4"
    assert response.headers["X-Model-Used"] == "Student x4"

    out_bytes = response.content
    out_img = Image.open(io.BytesIO(out_bytes))
    out_img.verify()

    out_img = Image.open(io.BytesIO(out_bytes))
    assert out_img.mode == "RGB"
    assert out_img.size == (64, 64)

    out_np = np.array(out_img)
    assert out_np.shape == (64, 64, 3)
    assert out_np.min() >= 0 and out_np.max() <= 255
    # Assert non-corrupt output: image pixel mean should reflect input intensity, not dark/black garbage
    assert out_np.mean() > 30.0

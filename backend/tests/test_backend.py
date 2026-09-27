"""Unit and Integration Tests for FastAPI Backend.

Author: Antigravity
Purpose: Validate FastAPI routes, status codes, scale validation, and enhance endpoint logic.
"""

import sys
import io
from pathlib import Path
import pytest
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
    assert "2x" in data["supported_scales"]


def test_health_endpoint():
    """Test health check GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert 2 in data["supported_scales"]


def test_enhance_endpoint_invalid_scale():
    """Test POST /enhance with an invalid scale parameter (e.g. scale=3)."""
    img = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/enhance",
        files={"file": ("test.png", buf, "image/png")},
        data={"scale": 3},
    )
    assert response.status_code == 400
    assert "Unsupported scale factor" in response.json()["detail"]


def test_enhance_endpoint_scale_2():
    """Test POST /enhance with scale=2."""
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/enhance",
        files={"file": ("test.png", buf, "image/png")},
        data={"scale": 2},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # Verify enhanced image dimensions (64x64)
    out_img = Image.open(io.BytesIO(response.content))
    assert out_img.size == (64, 64)


def test_enhance_endpoint_scale_4():
    """Test POST /enhance with scale=4."""
    img = Image.new("RGB", (16, 16), color="green")
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

    out_img = Image.open(io.BytesIO(response.content))
    assert out_img.size == (64, 64)

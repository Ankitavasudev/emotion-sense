"""
API tests for Emotion Sense
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Emotion Sense API"


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_analyze():
    """Test analyze endpoint."""
    response = client.post(
        "/api/v1/analyze",
        json={"text": "I love this amazing product!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "emotion" in data["data"]
    assert "confidence" in data["data"]


def test_analyze_negative():
    """Test analyze with negative text."""
    response = client.post(
        "/api/v1/analyze",
        json={"text": "This is terrible and awful!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["emotion"] in ["angry", "sad"]


def test_batch_analyze():
    """Test batch analysis."""
    response = client.post(
        "/api/v1/analyze/batch",
        json={
            "texts": [
                "I love this!",
                "This is terrible!",
                "It's okay."
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["results"]) == 3


def test_history():
    """Test history endpoint."""
    # First add some data
    client.post("/api/v1/analyze", json={"text": "Test message"})
    
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data


def test_stats():
    """Test stats endpoint."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_analyses" in data["data"]

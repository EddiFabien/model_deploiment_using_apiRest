"""
Tests for the OptionClass API.
"""
import json
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

# Test client
client = TestClient(app)

# Sample test data - Updated to match the exact schema expected by the API
SAMPLE_INPUT = {
    # Required fields with their expected types
    "sexe": 1.0,  # 0: Female, 1: Male
    "nb_frère": 2.0,  # Number of brothers
    "nb_sœur": 1.0,  # Number of sisters
    "commune_d_origine": 25,  # Origin commune code
    "habite_avec_les_parents": 1,  # 0: No, 1: Yes
    "electricite": 1,  # 0: No, 1: Yes
    "conn_sur_les_options": 1,  # 0: No, 1: Yes
    "MLG": 15.0,  # Malagasy language grade (0-20)
    "FRS": 14.5,  # French language grade (0-20)
    "ANG": 16.0,  # English language grade (0-20)
    "HG": 13.5,  # History-Geography grade (0-20)
    "SES": 12.0,  # Socio-Economic Studies grade (0-20)
    "MATHS": 18.0,  # Mathematics grade (0-20)
    "PC": 17.5,  # Physics-Chemistry grade (0-20)
    "SVT": 16.5,  # Life and Earth Sciences grade (0-20)
    "EPS": 15.0,  # Physical Education grade (0-20)
    "première_S": 14.5,  # First year grade
    "deuxième_S": 15.5,  # Second year grade
    "MOY_AN": 15.0  # Annual average grade
}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data
    assert "model_features" in data

def test_predict_endpoint():
    """Test the prediction endpoint with valid input."""
    response = client.post("/api/v1/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence_scores" in data
    assert "status" in data
    assert data["status"] == "success"
    assert data["prediction"] in ["L", "S", "OSE"]

def test_invalid_input():
    """Test the prediction endpoint with invalid input."""
    invalid_input = SAMPLE_INPUT.copy()
    invalid_input["sexe"] = 2.0  # Invalid value (should be 0 or 1)
    
    response = client.post("/api/v1/predict", json=invalid_input)
    assert response.status_code == 422  # Validation error

def test_model_features():
    """Test the model features endpoint."""
    response = client.get("/api/v1/model/features")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert isinstance(data["features"], list)

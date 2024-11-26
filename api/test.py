from fastapi.testclient import TestClient
import pytest
from main import app
import json
import numpy as np

client = TestClient(app)

# Test Data
valid_input = {
    "Age": 25,
    "Sleep_Quality": 7.5,
    "Daily_Steps": 8000,
    "Calories_Burned": 2500,
    "Heart_Rate": 75,
    "Sleep_Duration": 7.5,
    "Physical_Activity_Level": "medium",
    "Medication_Usage": "no",
    "Social_Interaction": 3
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] == True

def test_valid_prediction():
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert "anxiety_level" in data
    assert "depression_level" in data
    assert "prediction_details" in data
    
    # Check value ranges
    assert 0 <= data["anxiety_level"] <= 1
    assert 0 <= data["depression_level"] <= 1
    
    # Check prediction details
    assert data["prediction_details"]["anxiety_risk"] in ["low", "medium", "high"]
    assert data["prediction_details"]["depression_risk"] in ["low", "medium", "high"]

def test_invalid_age():
    invalid_input = valid_input.copy()
    invalid_input["Age"] = 150  # Age > 100
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_invalid_sleep_quality():
    invalid_input = valid_input.copy()
    invalid_input["Sleep_Quality"] = 11  # Sleep_Quality > 10
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_invalid_daily_steps():
    invalid_input = valid_input.copy()
    invalid_input["Daily_Steps"] = 12000  # Daily_Steps > 11000
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_invalid_activity_level():
    invalid_input = valid_input.copy()
    invalid_input["Physical_Activity_Level"] = "very high"  # Invalid category
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_invalid_medication_usage():
    invalid_input = valid_input.copy()
    invalid_input["Medication_Usage"] = "maybe"  # Invalid category
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_invalid_social_interaction():
    invalid_input = valid_input.copy()
    invalid_input["Social_Interaction"] = 6  # Social_Interaction > 5
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_missing_field():
    invalid_input = valid_input.copy()
    del invalid_input["Age"]  # Remove required field
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_prediction_consistency():
    """Test if the same input gives consistent predictions"""
    response1 = client.post("/predict", json=valid_input)
    response2 = client.post("/predict", json=valid_input)
    
    assert response1.json()["anxiety_level"] == response2.json()["anxiety_level"]
    assert response1.json()["depression_level"] == response2.json()["depression_level"]

def test_edge_case_inputs():
    edge_cases = [
        {
            # Minimum values
            "Age": 18,
            "Sleep_Quality": 1.0,
            "Daily_Steps": 0,
            "Calories_Burned": 0,
            "Heart_Rate": 60,
            "Sleep_Duration": 0,
            "Physical_Activity_Level": "low",
            "Medication_Usage": "no",
            "Social_Interaction": 0
        },
        {
            # Maximum values
            "Age": 100,
            "Sleep_Quality": 10.0,
            "Daily_Steps": 11000,
            "Calories_Burned": 2900,
            "Heart_Rate": 100,
            "Sleep_Duration": 24,
            "Physical_Activity_Level": "high",
            "Medication_Usage": "yes",
            "Social_Interaction": 5
        }
    ]
    
    for case in edge_cases:
        response = client.post("/predict", json=case)
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["anxiety_level"] <= 1
        assert 0 <= data["depression_level"] <= 1

def test_stress_test():
    """Test multiple rapid requests"""
    responses = []
    for _ in range(10):
        response = client.post("/predict", json=valid_input)
        responses.append(response)
    
    assert all(response.status_code == 200 for response in responses)

# Run the tests
if __name__ == "__main__":
    pytest.main(["-v"])
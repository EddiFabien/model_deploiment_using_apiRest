"""
Script to test the prediction endpoint.
"""
import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000/api/v1"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_model_features():
    """Test the model features endpoint."""
    print("Testing model features...")
    response = requests.get(f"{BASE_URL}/model/features")
    print(f"Status code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_predict():
    """Test the prediction endpoint."""
    print("Testing prediction endpoint...")
    
    # Sample input data - this should match the expected schema
    sample_input = {
        "sexe": 1.0,
        "nb_frere": 2.0,
        "nb_soeur": 1.0,
        "commune_origine": 25,
        "habite_avec_parents": 1,
        "electricite": 1,
        "conn_sur_options": 1,
        "MLG": 15.0,
        "FRS": 14.5,
        "ANG": 16.0,
        "HG": 13.5,
        "SES": 12.0,
        "MATHS": 18.0,
        "PC": 17.5,
        "SVT": 16.5,
        "EPS": 15.0,
        "premiere_S": 14.5,
        "deuxieme_S": 15.5,
        "MOY_AN": 15.0
    }
    
    print(f"Sending input: {json.dumps(sample_input, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_input,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status code: {response.status_code}")
        
        try:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except ValueError:
            print(f"Response text: {response.text}")
            
    except Exception as e:
        print(f"Error making request: {e}")
    
    print("-" * 50)

if __name__ == "__main__":
    test_health_check()
    test_model_features()
    test_predict()

from fastapi.testclient import TestClient
from main import app  # Note: If your main Python file is named 'app.py', change this to 'from main import app'

# This creates a fake "browser" to ping your API locally
client = TestClient(app)

def test_api_health_check():
    """Test if the API successfully boots up and returns a 200 OK status"""
    response = client.get("/")
    
    # The 'assert' keyword is the actual test. It says: "This MUST be true, or else fail the test."
    assert response.status_code == 200
    

def test_price_prediction():
    """Test if the ML model successfully returns a predicted price given valid input"""
    
    # 1. The fake package data we are sending to your API
    payload = {
        "weight_kg": 10.5,
        "distance_km": 500.0,
        "is_express": 1,
        "is_fragile": 0  #  (Use 0 for standard, 1 for fragile)
    }
    
    # 2. We use client.post because we are SENDING data to the prediction endpoint
    # (Note: Change "/predict" if your actual endpoint URL is named differently in your main.py)
    response = client.post("/predict", json=payload)
    
    # 3. Check if the server successfully processed the math (Status 200)
    assert response.status_code == 200
    
    # 4. Check if the API actually returned a price in the dictionary
    data = response.json()
    
    # (Note: Change "predicted_price" if your API returns the number under a different label)
    assert "predicted_shipping_cost_usd" in data
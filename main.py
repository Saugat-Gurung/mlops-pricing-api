from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the web application
app = FastAPI(
    title="Logistics Pricing API",
    description="An MLOps endpoint for dynamic shipping cost prediction."
)

# 2. Load the trained machine learning model into memory
model = joblib.load("pricing_model.joblib")

# 3. Define the strict data structure we expect from users
class ShippingRequest(BaseModel):
    weight_kg: float
    distance_km: float
    is_express: int
    is_fragile: int

# Create a welcome page so we don't get a 404 error
@app.get("/")
def read_root():
    return {"message": "Welcome to the Logistics Pricing API. Go to /docs to test the engine."}

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_price(request: ShippingRequest):
    # Convert the incoming JSON data into a pandas DataFrame (what the model expects)
    input_data = pd.DataFrame([{
        "weight_kg": request.weight_kg,
        "distance_km": request.distance_km,
        "is_express": request.is_express,
        "is_fragile": request.is_fragile
    }])

    # Ask the model for a prediction
    prediction = model.predict(input_data)

    # Return the formatted result as a JSON response
    return {
        "predicted_shipping_cost_usd": round(float(prediction[0]), 2)
    }
    
    # A GET endpoint to simulate tracking a package
@app.get("/tracking/{tracking_number}")
def get_shipment_status(tracking_number: str):
    # In a real production environment, this would query a SQL database.
    # Here, we return a dynamic mock response using the user's input.
    return {
        "tracking_id": tracking_number,
        "current_location": "London Distribution Center",
        "status": "In Transit",
        "estimated_delivery": "Tomorrow by 8:00 PM",
        "update_history": [
            "Package scanned at origin facility",
            "Departed regional sorting hub",
            "Arrived at London Distribution Center"
        ]
    }
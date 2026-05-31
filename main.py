from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Logistics Pricing Predictor</title>
        <style>
            body { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .container { background-color: #161b22; padding: 40px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 8px 24px rgba(0,0,0,0.5); width: 100%; max-width: 420px; }
            h2 { text-align: center; margin-top: 0; color: #58a6ff; font-weight: 600; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-size: 14px; font-weight: 500; }
            input, select { width: 100%; padding: 12px; background-color: #0d1117; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; box-sizing: border-box; font-size: 14px; outline: none; }
            input:focus, select:focus { border-color: #58a6ff; }
            button { width: 100%; padding: 14px; background-color: #238636; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background-color 0.2s; margin-top: 10px; }
            button:hover { background-color: #2ea043; }
            .result-box { margin-top: 25px; padding: 15px; background-color: #0d1117; border-left: 4px solid #58a6ff; border-radius: 6px; display: none; font-size: 18px; text-align: center; font-weight: 600; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>AI Pricing Predictor</h2>
            <form id="predictForm">
                <div class="form-group">
                    <label>Weight (kg)</label>
                    <input type="number" id="weight" step="0.1" required placeholder="e.g. 73">
                </div>
                <div class="form-group">
                    <label>Distance (km)</label>
                    <input type="number" id="distance" step="1" required placeholder="e.g. 6500">
                </div>
                <div class="form-group">
                    <label>Shipping Speed</label>
                    <select id="express">
                        <option value="0">Standard</option>
                        <option value="1">Express</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Is it Fragile?</label>
                    <select id="fragile">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <button type="button" onclick="calculatePrice()">Get Shipping Cost</button>
            </form>
            
            <div class="result-box" id="resultBox">
                Estimated Cost: <span id="priceText" style="color: #58a6ff;"></span>
            </div>
        </div>

        <script>
            async function calculatePrice() {
                const weight = parseFloat(document.getElementById('weight').value);
                const distance = parseFloat(document.getElementById('distance').value);
                const express = parseInt(document.getElementById('express').value);
                const fragile = parseInt(document.getElementById('fragile').value);

                if (!weight || !distance) {
                    alert("Please fill in all numerical fields.");
                    return;
                }

                const button = document.querySelector('button');
                button.innerText = "Calculating...";

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            "weight_kg": weight,
                            "distance_km": distance,
                            "is_express": express,
                            "is_fragile": fragile
                        })
                    });

                    const data = await response.json();
                    
                    // Grabs the exact prediction value from your API's dictionary
                    let price = data.predicted_price || data.price || data.prediction || Object.values(data)[0]; 
                    
                    if(typeof price === 'number') {
                        price = '£' + price.toFixed(2);
                    } 

                    document.getElementById('priceText').innerText = price;
                    document.getElementById('resultBox').style.display = 'block';
                } catch (error) {
                    alert("Error connecting to the AI model.");
                } finally {
                    button.innerText = "Get Shipping Cost";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
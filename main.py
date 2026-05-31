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
            .container { background-color: #161b22; padding: 40px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 8px 24px rgba(0,0,0,0.5); width: 100%; max-width: 480px; }
            h2 { text-align: center; margin-top: 0; color: #58a6ff; font-weight: 600; }
            .form-row { display: flex; gap: 15px; margin-bottom: 20px; }
            .form-group { flex: 1; margin-bottom: 20px; }
            .form-row .form-group { margin-bottom: 0; }
            label { display: block; margin-bottom: 8px; font-size: 14px; font-weight: 500; }
            input, select { width: 100%; padding: 12px; background-color: #0d1117; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; box-sizing: border-box; font-size: 14px; outline: none; }
            input:focus, select:focus { border-color: #58a6ff; }
            button { width: 100%; padding: 14px; background-color: #238636; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background-color 0.2s; margin-top: 10px; }
            button:hover { background-color: #2ea043; }
            .result-box { margin-top: 25px; padding: 15px; background-color: #0d1117; border-left: 4px solid #58a6ff; border-radius: 6px; display: none; font-size: 15px; line-height: 1.6; }
            .highlight { color: #58a6ff; font-weight: bold; font-size: 18px;}
            .detail { color: #8b949e; font-size: 13px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Global Freight Estimator</h2>
            <form id="predictForm">
                <div class="form-group">
                    <label>Package Weight (kg)</label>
                    <input type="number" id="weight" step="0.1" required placeholder="e.g. 73">
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Origin Hub</label>
                        <select id="origin">
                            <option value="51.5074,-0.1278">London, UK</option>
                            <option value="40.7128,-74.0060">New York, USA</option>
                            <option value="35.6762,139.6503">Tokyo, Japan</option>
                            <option value="-33.8688,151.2093">Sydney, Australia</option>
                            <option value="25.2048,55.2708">Dubai, UAE</option>
                            <option value="48.8566,2.3522">Paris, France</option>
                            <option value="1.3521,103.8198">Singapore</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Destination Hub</label>
                        <select id="destination">
                            <option value="40.7128,-74.0060">New York, USA</option>
                            <option value="51.5074,-0.1278">London, UK</option>
                            <option value="35.6762,139.6503">Tokyo, Japan</option>
                            <option value="-33.8688,151.2093">Sydney, Australia</option>
                            <option value="25.2048,55.2708">Dubai, UAE</option>
                            <option value="48.8566,2.3522">Paris, France</option>
                            <option value="1.3521,103.8198">Singapore</option>
                        </select>
                    </div>
                </div>
                <div class="form-group">
                    <label>Transport Method</label>
                    <select id="express">
                        <option value="0">Standard (Train / Cargo Ship / Trucks)</option>
                        <option value="1">Express (Airway)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Is it Fragile?</label>
                    <select id="fragile">
                        <option value="0">No, Standard Packaging</option>
                        <option value="1">Yes, Handle With Care</option>
                    </select>
                </div>
                <button type="button" onclick="calculatePrice()">Calculate Route & Price</button>
            </form>
            
            <div class="result-box" id="resultBox">
                <div>Estimated Cost: <span class="highlight" id="priceText"></span></div>
                <div style="margin-top: 10px;">Estimated Transit Time: <span class="highlight" id="timeText" style="color: #3fb950;"></span></div>
                <div class="detail" style="margin-top: 10px;" id="distanceText"></div>
            </div>
        </div>

        <script>
            // Haversine formula to calculate distance between two coordinates
            function getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2) {
                const R = 6371; // Radius of the earth in km
                const dLat = (lat2 - lat1) * Math.PI / 180;
                const dLon = (lon2 - lon1) * Math.PI / 180;
                const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                          Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon/2) * Math.sin(dLon/2);
                const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
                return Math.round(R * c);
            }

            async function calculatePrice() {
                const weight = parseFloat(document.getElementById('weight').value);
                const originCoords = document.getElementById('origin').value.split(',');
                const destCoords = document.getElementById('destination').value.split(',');
                const express = parseInt(document.getElementById('express').value);
                const fragile = parseInt(document.getElementById('fragile').value);

                if (!weight) {
                    alert("Please enter a package weight.");
                    return;
                }

                // Extract Lat/Lon and calculate real world distance
                const lat1 = parseFloat(originCoords[0]), lon1 = parseFloat(originCoords[1]);
                const lat2 = parseFloat(destCoords[0]), lon2 = parseFloat(destCoords[1]);
                const distance = getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2);

                if (distance === 0) {
                    alert("Origin and Destination cannot be the same hub.");
                    return;
                }

                // Calculate Delivery Time roughly based on transport speeds
                let days = 0;
                if (express === 1) {
                    days = Math.ceil(distance / 2500) + 1; // Approx airplane speed + handling
                } else {
                    days = Math.ceil(distance / 800) + 4; // Approx truck/cargo speed + handling
                }

                const button = document.querySelector('button');
                button.innerText = "Routing...";

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
                    let price = data.predicted_shipping_cost_usd || data.predicted_price || Object.values(data)[0]; 
                    
                    // Convert USD to GBP for a UK-centric portfolio
                    if(typeof price === 'number') {
                        price = '£' + (price * 0.79).toFixed(2);
                    } 

                    document.getElementById('priceText').innerText = price;
                    document.getElementById('timeText').innerText = days + (days === 1 ? " Day" : " Days");
                    document.getElementById('distanceText').innerText = "Route distance calculated at " + distance.toLocaleString() + " km.";
                    
                    document.getElementById('resultBox').style.display = 'block';
                } catch (error) {
                    alert("Error connecting to the AI model.");
                } finally {
                    button.innerText = "Calculate Route & Price";
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
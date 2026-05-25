# MLOps Dynamic Pricing API 📦

An end-to-end Machine Learning Operations (MLOps) project that bridges the gap between data science and software engineering. This project trains a Random Forest regression model on supply chain data and deploys it as a high-speed, containerized web API.

### Architecture & Tech Stack
* **Machine Learning:** Scikit-Learn (Random Forest Regressor), Pandas, NumPy
* **API Framework:** FastAPI, Pydantic, Uvicorn
* **Deployment:** Docker
* **Model Serialization:** Joblib

### Features
* **Dynamic Pricing Engine:** Calculates real-time shipping costs based on package weight, delivery distance, and shipping tiers (Express/Fragile).
* **RESTful Endpoints:** * `POST /predict`: Ingests JSON payload and returns the ML model's mathematical prediction.
  * `GET /tracking/{id}`: Simulates a database lookup for package routing statuses.
* **Strict Data Validation:** Utilizes Pydantic to ensure the ML model never crashes from bad user inputs.
* **Fully Containerized:** Wrapped in a Docker image for universal, cross-platform deployment.

### How to Run Locally using Docker
1. Clone the repository.
2. Ensure Docker Desktop is running.
3. Build the image: `docker build -t logistics-pricing-api .`
4. Run the container: `docker run -p 8000:8000 logistics-pricing-api`
5. Navigate to `http://localhost:8000/docs` to interact with the live model.
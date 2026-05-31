# MLOps Dynamic Pricing API 📦

An end-to-end Machine Learning Operations (MLOps) project that bridges the gap between data science and software engineering. This project trains a Random Forest regression model on supply chain data and deploys it as a high-speed, containerized web API.

**Live Demo:** [Test the live AI pricing engine here] https://mlops-pricing-api.onrender.com

### Recent Deployments & UI Upgrades:
* **Cloud Infrastructure:** Engineered an automated CI/CD pipeline triggering instant Docker container deployments to the cloud.
* **Geospatial Routing:** Replaced static distance inputs with an active routing system. It utilizes the Haversine formula to calculate global hub-to-hub flight distances using raw latitude/longitude coordinates.
* **Full-Stack Integration:** Built a custom HTML/JS frontend directly into the Python backend to translate mathematical outputs into real-world GBP (£) and estimated shipping timelines.

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
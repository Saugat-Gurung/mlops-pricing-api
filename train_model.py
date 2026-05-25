import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

print("Loading logistics data...")
# 1. Load the data
df = pd.read_csv("logistics_data.csv")

# 2. Separate the inputs (features) from the output (target)
X = df[['weight_kg', 'distance_km', 'is_express', 'is_fragile']]
y = df['shipping_cost']

# 3. Split the data into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the model
print("Training the Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test the model to see how accurate it is
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Model trained successfully!")
print(f"Mean Absolute Error: ${error:.2f} (On average, our predictions are off by this amount)")

# 6. Save the trained model to a file
print("Saving model to disk...")
joblib.dump(model, "pricing_model.joblib")
print("✅ Saved as pricing_model.joblib")
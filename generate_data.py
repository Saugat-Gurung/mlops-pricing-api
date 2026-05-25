import pandas as pd
import numpy as np

# Set a random seed so the data is the same every time we run it
np.random.seed(42)

# Generate 5000 synthetic shipping records
n_samples = 5000

data = {
    "weight_kg": np.random.uniform(0.5, 50.0, n_samples),
    "distance_km": np.random.uniform(5.0, 3000.0, n_samples),
    # 0 for Standard, 1 for Express
    "is_express": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    # 0 for Normal, 1 for Fragile
    "is_fragile": np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# Calculate the actual pricing logic (the "ground truth" our AI will try to learn)
# Base cost is $5.00
# $0.50 per kg
# $0.01 per km
# Express shipping doubles the distance cost
# Fragile adds a flat $15 handling fee
# Plus some random real-world noise (traffic, fuel surcharges, etc.)

df['shipping_cost'] = (
    5.0 + 
    (df['weight_kg'] * 0.50) + 
    (df['distance_km'] * 0.01 * np.where(df['is_express'] == 1, 2.0, 1.0)) + 
    (df['is_fragile'] * 15.0) +
    np.random.normal(0, 2.5, n_samples) # Adding random noise
)

# Ensure no negative prices due to the noise
df['shipping_cost'] = df['shipping_cost'].clip(lower=5.0)

# Round to 2 decimal places like real currency
df = df.round(2)

# Save it to a CSV
df.to_csv("logistics_data.csv", index=False)
print("✅ Successfully generated logistics_data.csv with 5000 records!")
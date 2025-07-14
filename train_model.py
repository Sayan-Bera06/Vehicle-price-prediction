# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/vehicles.csv")  # ✅ Make sure the file exists

# Drop irrelevant columns
df.drop(columns=['name', 'description'], inplace=True, errors='ignore')

# Convert 'year' to 'vehicle_age'
if 'year' in df.columns:
    df['vehicle_age'] = 2025 - df['year']
    df.drop(columns=['year'], inplace=True)

# Fill missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

df[numeric_cols] = df[numeric_cols].fillna(0)
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_cols)

# Scale 'mileage' and 'vehicle_age'
scaler = StandardScaler()
# Fit scaler on both columns at once
scaler = StandardScaler()
df[['mileage', 'vehicle_age']] = scaler.fit_transform(df[['mileage', 'vehicle_age']])


# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Separate features and target
if 'price' not in df.columns:
    raise ValueError("The dataset must contain a 'price' column.")

X = df.drop('price', axis=1)
y = df['price']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/vehicle_model.pkl")

# Evaluation
y_pred = model.predict(X_test)
print(f"✅ R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"✅ MSE: {mean_squared_error(y_test, y_pred):,.2f}")

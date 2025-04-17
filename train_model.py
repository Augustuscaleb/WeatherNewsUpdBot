import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load preprocessed data
file_path = "C:/Users/user/Desktop/WeatherNewsBot/preprocessed_weather_data.csv"
data = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")

# Select only required columns
data = data.dropna(subset=["Temperature", "Humidity", "Pressure", "Wind Speed"])
X = data[["Year", "Month", "Day", "Hour"]]  # Features
y = data[["Temperature", "Humidity", "Pressure", "Wind Speed"]]  # Targets

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")

# Print evaluation results
print("Model Performance:")
for i, column in enumerate(y.columns):
    print(f"{column} MSE: {mse[i]:.4f}")

# Save trained model
model_path = "C:/Users/user/Desktop/WeatherNewsBot/weather_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

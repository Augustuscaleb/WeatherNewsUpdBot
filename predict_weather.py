import pandas as pd
import joblib

# Load trained model
model_path = "C:/Users/user/Desktop/WeatherNewsBot/weather_model.pkl"
model = joblib.load(model_path)
print("Model loaded successfully!")

# Example input for prediction
new_data = pd.DataFrame({
    "Year": [2025],
    "Month": [3],
    "Day": [15],
    "Hour": [12]
})

# Make predictions
predictions = model.predict(new_data)
predicted_df = pd.DataFrame(predictions, columns=["Temperature", "Humidity", "Pressure", "Wind Speed"])

# Convert Temperature from Kelvin to Celsius
predicted_df["Temperature"] = predicted_df["Temperature"] - 273.15

# Print predictions
print("Predicted Weather Conditions:")
print(predicted_df)

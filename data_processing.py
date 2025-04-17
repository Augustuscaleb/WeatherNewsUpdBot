import pandas as pd
import os

# Define folder path
folder_path = "C:/Users/user/Desktop/WeatherNewsBot"

# Load relevant CSVs
temperature_df = pd.read_csv(os.path.join(folder_path, "temperature.csv"), index_col=0)
humidity_df = pd.read_csv(os.path.join(folder_path, "humidity.csv"), index_col=0)
pressure_df = pd.read_csv(os.path.join(folder_path, "pressure.csv"), index_col=0)
wind_speed_df = pd.read_csv(os.path.join(folder_path, "wind_speed.csv"), index_col=0)

# Combine datasets
combined_df = pd.concat(
    [temperature_df.iloc[:, 0], humidity_df.iloc[:, 0], pressure_df.iloc[:, 0], wind_speed_df.iloc[:, 0]],
    axis=1, join="inner"
)

# Rename columns
combined_df.columns = ["Temperature", "Humidity", "Pressure", "Wind Speed"]

# Convert index to datetime
combined_df.index = pd.to_datetime(combined_df.index)

# Handle missing values
combined_df = combined_df.fillna(method="ffill").interpolate()

# Add time-based features
combined_df["Year"] = combined_df.index.year
combined_df["Month"] = combined_df.index.month
combined_df["Day"] = combined_df.index.day
combined_df["Hour"] = combined_df.index.hour

# Save preprocessed data
preprocessed_path = os.path.join(folder_path, "preprocessed_weather_data.csv")
combined_df.to_csv(preprocessed_path)
print(f"Preprocessed data saved to {preprocessed_path}")

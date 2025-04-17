from telegram import Update
from telegram.ext import ContextTypes
import pandas as pd
import joblib

# Load the model
model_path = "C:/Users/user/Desktop/WeatherNewsBot/weather_model.pkl"
model = joblib.load(model_path)

# Define prediction function
async def predict_weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        args = context.args
        if len(args) != 4:
            await update.message.reply_text("Please provide 4 values: Year, Month, Day, Hour.")
            return

        year, month, day, hour = map(int, args)
        new_data = pd.DataFrame({"Year": [year], "Month": [month], "Day": [day], "Hour": [hour]})
        predictions = model.predict(new_data)
        temperature, humidity, pressure, wind_speed = predictions[0]

        await update.message.reply_text(
            f"Predicted Weather:\n"
            f"Temperature: {temperature:.2f}°C\n"
            f"Humidity: {humidity:.2f}%\n"
            f"Pressure: {pressure:.2f} hPa\n"
            f"Wind Speed: {wind_speed:.2f} m/s"
        )
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

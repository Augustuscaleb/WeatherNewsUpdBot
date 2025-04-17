import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes,  MessageHandler, filters
import logging
import requests
import pandas as pd
import joblib
from datetime import datetime
import openai
import difflib  # For spell-checking
from geopy.geocoders import Nominatim
import numpy as np
import predict_weather


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load API keys from .env file
load_dotenv()
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_API_KEY:
    logger.error("Missing TELEGRAM_API_KEY. The bot will not function.")
    exit(1)

if not WEATHER_API_KEY:
    logger.warning("Missing WEATHER_API_KEY. The /weather command will not work.")

if not NEWS_API_KEY:
    logger.warning("Missing NEWS_API_KEY. The /news command will not work.")

if not OPENAI_API_KEY:
    logger.warning("Missing OPENAI_API_KEY. AI responses will not be available.")


openai.api_key = OPENAI_API_KEY

# API URLs
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
NEWS_API_URL = "https://newsdata.io/api/1/news"

# Load the prediction model
model_path = "C:/Users/user/Desktop/WeatherNewsBot/weather_model.pkl"
model = None
try:
    model = joblib.load(model_path)
    logger.info("Prediction model loaded successfully!")
    geolocator = Nominatim(user_agent="weather_bot")

except Exception as e:
    logger.error(f"Error loading prediction model: {e}")

# Helper function: Spell-checking
def correct_spelling(query, valid_words):
    closest_match = difflib.get_close_matches(query, valid_words, n=1, cutoff=0.7)
    return closest_match[0] if closest_match else None

# Helper function: Deduplicate news articles
def deduplicate_articles(articles):
    seen_titles = set()
    unique_articles = []
    for article in articles:
        title = article.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)
    return unique_articles

# Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌟 Hello there! I'm your Weather and News Bot! 🌦️📰\n\n"
        "I'm here to keep you updated on the latest weather and news. Here's what I can do for you:\n\n"
        "☀️ Check the weather:\n"
        "Just type: `/weather <city or country>`\n"
        "📌 Example: `/weather Accra`\n\n"
        "📰 Get the latest news:\n"
        "Use: `/news <country or keyword>`\n"
        "📌 Example: `/news Ghana`\n\n"
        "🌍 Predict future weather conditions:\n"
        "Try: `/predict <year> <month> <day> <hour>`\n"
        "📌 Example: `/predict 2025 3 15 12`\n\n"
        "💬 Feel free to request the weather, news, or make weather predictions and I'll do my best to help! Let's get started! 🚀"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Here are the commands you can use:\n"
        "/start - Start the bot\n"
        "/weather <city or country> - Get current weather updates for a city\n"
        "/news <country or city> - Get the latest news\n"
        "/predict <year> <month> <day> <hour> - Predict future weather\n"
        "/help - Show this help message"
    )

async def weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide a city name. \n Example: /weather London")
        return

    city_name = " ".join(context.args)
    try:
        response = requests.get(
            WEATHER_API_URL,
            params={"q": city_name, "appid": WEATHER_API_KEY, "units": "metric"},
        )
        data = response.json()

        if data.get("cod") != 200:
            await update.message.reply_text(f"Error: {data.get('message', 'Unable to fetch weather data.')}")
            return

        weather_desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        # Clothing & Activity Advice
        if temp > 30:
            advice = "It's quite hot! ☀️ Stay hydrated, wear light clothes, and apply sunscreen. A hat or sunglasses would be a good choice! 🧢😎"
        elif 20 <= temp <= 30:
            advice = "The weather is pleasant! 😊 You can wear light clothing, but carrying a light jacket might be a good idea if you're staying out late."
        elif 10 <= temp < 20:
            advice = "It's a bit chilly! 🍃 A sweater or a light jacket is recommended. Perfect weather for a warm coffee or tea! ☕"
        elif 0 <= temp < 10:
            advice = "It's cold outside! 🥶 Wear a thick jacket, gloves, and a scarf. A hot drink will keep you cozy! ❄️☕"
        else:
            advice = "It's freezing! 🧊 Dress warmly with layers, gloves, and a hat. Stay indoors if possible and keep yourself warm! 🔥"

        if "rain" in weather_desc.lower():
            advice += " Also, don't forget to take an umbrella or a raincoat! ☔🌧️"
        elif "snow" in weather_desc.lower():
            advice += " It's snowy outside! ⛄ Wear waterproof boots and be careful on slippery roads. ❄️"
        elif "wind" in weather_desc.lower():
            advice += " It's quite windy! 💨 Hold onto your hat and be mindful of flying debris."

        await update.message.reply_text(
            f"🌍 Weather in {city_name}:\n"
            f"- 🌦️ {weather_desc}\n"
            f"- 🌡️ Temperature: {temp}°C (Feels like {feels_like}°C)\n"
            f"- 💧 Humidity: {humidity}%\n"
            f"- 🌬️ Wind Speed: {wind_speed} m/s\n\n"
            f"🔹 Advice: {advice}"
        )

    except Exception as e:
        logger.error(f"Weather Command Error: {e}")
        await update.message.reply_text("Failed to fetch weather data. Please try again later.")



async def predict_weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        args = context.args
        if len(args) != 4:
            await update.message.reply_text("Please provide 4 values: Year, Month, Day, Hour.\nExample: /predict 2025 3 15 12")
            return

        # Convert inputs to integers
        year, month, day, hour = map(int, args)

        # Prepare data for prediction
        new_data = pd.DataFrame({"Year": [year], "Month": [month], "Day": [day], "Hour": [hour]})

        # Make prediction
        predictions = model.predict(new_data)
        temperature, humidity, pressure, wind_speed = predictions[0]

        # Convert Temperature from Kelvin to Celsius
        temperature = temperature - 273.15

        # Generate weather description
        if temperature > 30:
            weather_desc = "It's likely to be very hot, stay hydrated! ☀️"
        elif 20 <= temperature <= 30:
            weather_desc = "The weather seems warm and pleasant. 🌤️"
        elif 10 <= temperature < 20:
            weather_desc = "It might be a bit chilly, grab a jacket. 🍂"
        else:
            weather_desc = "Expect cold weather. Keep warm! ❄️"

        if humidity > 80:
            weather_desc += " High humidity levels suggest it might feel muggy. 💧"
        elif humidity < 40:
            weather_desc += " Low humidity means it might feel dry. 🌵"

        if wind_speed > 10:
            weather_desc += " It could be windy, be cautious. 🌬️"

        # Send prediction response
        await update.message.reply_text(
            f"Predicted Weather on {year}-{month:02d}-{day:02d} at {hour}:00:\n"
            f"- Temperature: {temperature:.2f}°C\n"
            f"- Humidity: {humidity:.2f}%\n"
            f"- Pressure: {pressure:.2f} hPa\n"
            f"- Wind Speed: {wind_speed:.2f} m/s\n\n"
            f"{weather_desc}"
        )

    except Exception as e:
        logger.error(f"Predict Command Error: {e}")
        await update.message.reply_text(f"Error: {e}")


# Helper function
def generate_weather_description(temperature, humidity, wind_speed):
    desc = "Weather Summary: "
    if temperature > 30:
        desc += "Hot weather expected. ☀️"
    elif 20 <= temperature <= 30:
        desc += "Mild and warm. 🌤️"
    else:
        desc += "Cool temperatures. ❄️"

    if humidity > 80:
        desc += " High humidity. 💧"
    if wind_speed > 10:
        desc += " Windy conditions. 🌬️"

    return desc

async def handle_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide a keyword or country name. Example: /news Ghana or /news USA.")
        return

    query = " ".join(context.args).strip().lower()
    valid_queries = ["ghana", "usa", "technology", "sports", "business"]

    corrected_query = correct_spelling(query, valid_queries) or query
    await update.message.reply_text(f"Fetching news for '{corrected_query}'...")

    try:
        response = requests.get(
            NEWS_API_URL,
            params={"apikey": NEWS_API_KEY, "q": query, "language": "en"},
        )
        data = response.json()

        if response.status_code != 200:
            await update.message.reply_text(f"API Error: {data.get('message', 'Unable to fetch news.')}")
            return

        if data.get("status") != "success":
            await update.message.reply_text(f"API reported failure: {data}")
            return

        articles = data.get("results", [])
        articles = deduplicate_articles(articles)

        if not articles:
            await update.message.reply_text(
                f"No news articles found for '{query}'. Try a broader term like a country name.")
            return

        news_message = f"📰 Top News for '{query.capitalize()}':\n\n"
        for article in articles[:5]:
            title = article.get("title", "No Title")
            link = article.get("link", "#")
            source = article.get("source_id", "Unknown Source")
            date = article.get("pubDate", "Unknown Date")
            news_message += f"- [{title}]({link}) ({source}, {date})\n"

        await update.message.reply_text(news_message, parse_mode="Markdown", disable_web_page_preview=True)

    except Exception as e:
        logger.error(f"News Command Error: {e}")
        await update.message.reply_text(f"Error fetching news: {e}")


async def generate_ai_response(prompt):
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        return "Sorry, I couldn't process your request right now."


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.lower()

    # Define allowed keywords related to weather, news, or predictions
    allowed_keywords = ["weather", "news", "predict", "forecast", "temperature", "rain", "humidity", "storm"]

    # Check if the message contains relevant words
    if any(keyword in user_message for keyword in allowed_keywords):
        response = await generate_ai_response(user_message)
    else:
        response = (
            "I'm sorry, but I only provide weather updates, news, and weather predictions. 🌦️📰\n\n"
            "Try using:\n"
            "- /weather <city> (e.g., /weather London)\n"
            "- /news <country or keyword> (e.g., /news Ghana)\n"
            "- /predict <year> <month> <day> <hour> (e.g., /predict 2025 07 14 15)"
        )

    await update.message.reply_text(response)


# Main function
def main():
    application = Application.builder().token(TELEGRAM_API_KEY).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("weather", weather))
    application.add_handler(CommandHandler("news", handle_news_command))
    application.add_handler(CommandHandler("predict", predict_weather))

    # Default message handler for unrecognized commands or messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


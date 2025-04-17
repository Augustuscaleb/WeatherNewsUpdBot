from dotenv import load_dotenv
import os

load_dotenv()

print("Telegram API Key:", os.getenv("TELEGRAM_API_KEY"))
print("Weather API Key:", os.getenv("WEATHER_API_KEY"))
print("News API Key:", os.getenv("NEWS_API_KEY"))
print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))

import pandas as pd
import requests
import joblib
from datetime import datetime

API_KEY = 'a4f54718b17aa482e0b0a9f2e6220fc0'
WEATHER_CACHE = {}

# Helper to map month to season
SEASON_MAP = {1: 'Winter', 2: 'Winter', 12: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall'}

def fetch_weather(city, state, api_key=API_KEY):
    key = f"{city},{state}"
    if key in WEATHER_CACHE:
        return WEATHER_CACHE[key]
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},{state},US&limit=1&appid={api_key}"
    try:
        geo_resp = requests.get(geo_url)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data:
            return {'temperature': 20, 'humidity': 50, 'condition': 'Clear'}
        lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_resp = requests.get(weather_url)
        weather_resp.raise_for_status()
        data = weather_resp.json()
        weather = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'condition': data['weather'][0]['main']
        }
        WEATHER_CACHE[key] = weather
        return weather
    except Exception as e:
        print(f"Weather fetch error for {city}, {state}: {e}")
        return {'temperature': 20, 'humidity': 50, 'condition': 'Clear'}

def extract_season(month):
    return SEASON_MAP.get(month, 'Unknown')

def predict_discount(product_serial, city, state, order_date, segment, ship_mode, df, model):
    product = df[df['Product ID'] == product_serial]
    if product.empty:
        print(f"Product serial {product_serial} not found.")
        return None
    weather = fetch_weather(city, state)
    # Parse order date
    try:
        order_dt = pd.to_datetime(order_date, dayfirst=True)
    except Exception:
        print("Invalid order date format. Use YYYY-MM-DD.")
        return None
    order_month = order_dt.month
    order_day_of_week = order_dt.dayofweek
    season = extract_season(order_month)
    X_pred = pd.DataFrame({
        'Category': [product.iloc[0]['Category']],
        'Sub-Category': [product.iloc[0]['Sub-Category']],
        'Product ID': [product_serial],
        'Sales': [product.iloc[0]['Sales']],
        'City': [city],
        'State': [state],
        'Segment': [segment],
        'Ship Mode': [ship_mode],
        'order_month': [order_month],
        'order_day_of_week': [order_day_of_week],
        'season': [season],
        'temperature': [weather['temperature']],
        'humidity': [weather['humidity']],
        'condition': [weather['condition']]
    })
    discount = model.predict(X_pred)[0]
    return max(0, round(discount, 2))

if __name__ == "__main__":
    # Load model and data
    print("Loading model and data...")
    model = joblib.load('discount_model.joblib')
    df = pd.read_csv('train.csv')
    # User input
    serial = input("Enter product serial (Product ID): ")
    city = input("Enter city: ")
    state = input("Enter state: ")
    order_date = input("Enter order date (YYYY-MM-DD): ")
    segment = input("Enter customer segment (e.g., Consumer, Corporate, Home Office): ")
    ship_mode = input("Enter ship mode (e.g., First Class, Second Class, Standard Class, Same Day): ")
    discount = predict_discount(serial, city, state, order_date, segment, ship_mode, df, model)
    if discount is not None:
        print(f"Predicted discount for product {serial} in {city}, {state} on {order_date}: {discount}%") 
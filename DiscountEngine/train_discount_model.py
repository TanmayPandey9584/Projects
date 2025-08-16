import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

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

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    # Parse dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['order_month'] = df['Order Date'].dt.month
    df['order_day_of_week'] = df['Order Date'].dt.dayofweek
    df['season'] = df['order_month'].apply(extract_season)
    # Simulate discount for training
    np.random.seed(42)
    df['discount'] = (df['Sales'] / df['Sales'].max()) * 20 + np.random.normal(0, 2, len(df))
    # Fetch weather features
    weather_features = df.apply(lambda row: fetch_weather(row['City'], row['State']), axis=1)
    df['temperature'] = [w['temperature'] for w in weather_features]
    df['humidity'] = [w['humidity'] for w in weather_features]
    df['condition'] = [w['condition'] for w in weather_features]
    return df

def train_discount_model(df):
    features = [
        'Category', 'Sub-Category', 'Product ID', 'Sales',
        'City', 'State', 'Segment', 'Ship Mode',
        'order_month', 'order_day_of_week', 'season',
        'temperature', 'humidity', 'condition'
    ]
    X = df[features]
    y = df['discount']
    categorical = [
        'Category', 'Sub-Category', 'Product ID', 'City', 'State',
        'Segment', 'Ship Mode', 'season', 'condition'
    ]
    numeric = ['Sales', 'order_month', 'order_day_of_week', 'temperature', 'humidity']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numeric)
    ])
    model = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return model

if __name__ == "__main__":
    print("Loading and preparing data...")
    df = load_and_prepare_data('train.csv')
    print("Training model...")
    model = train_discount_model(df)
    joblib.dump(model, 'discount_model.joblib')
    print("Model trained and saved as discount_model.joblib.") 
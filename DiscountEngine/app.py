import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# 1. Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("discount_model.joblib")

model = load_model()

# -------------------------------------------------
# 2. Extract feature names dynamically
# -------------------------------------------------
def get_feature_names(model):
    pre = model.named_steps['pre']
    categorical = pre.transformers_[0][2]
    numeric = pre.transformers_[1][2]
    return categorical + numeric

features = get_feature_names(model)

# -------------------------------------------------
# 3. Build Input DataFrame
# -------------------------------------------------
def build_input(city, state, segment, ship_mode,
                order_month, order_day_of_week, season, temperature, humidity, condition):
    X_pred = pd.DataFrame({
        'City': [city],
        'State': [state],
        'Segment': [segment],
        'Ship Mode': [ship_mode],
        'order_month': [order_month],
        'order_day_of_week': [order_day_of_week],
        'season': [season],
        'temperature': [temperature],
        'humidity': [humidity],
        'condition': [condition],
        # 👉 Add defaults for any required features you’re not taking as input
        'Category': ['Furniture'],
        'Sub-Category': ['Chairs'],
        'Product ID': ['FUR-1234'],
        'Sales': [200.0]
    })
    return X_pred[features]

# -------------------------------------------------
# 4. Streamlit UI
# -------------------------------------------------
st.title("Discount Prediction App")
st.write("Provide details below to predict the discount:")

with st.form("prediction_form"):
    city = st.text_input("City", "New York")
    state = st.text_input("State", "NY")
    segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
    ship_mode = st.selectbox("Ship Mode", ["First Class", "Second Class", "Standard Class"])
    order_month = st.number_input("Order Month", 1, 12, 6)
    order_day_of_week = st.number_input("Order Day of Week", 0, 6, 2)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    temperature = st.number_input("Temperature (°C)", -10, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    condition = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Snowy"])

    submitted = st.form_submit_button("Predict Discount")

    if submitted:
        X_pred = build_input(city, state, segment, ship_mode,
                             order_month, order_day_of_week, season,
                             temperature, humidity, condition)
        discount = model.predict(X_pred)[0]
        st.metric("Predicted Discount", f"{discount:.2f} %")

# -------------------------------------------------
# 5. Footer (static signature)
# -------------------------------------------------
st.markdown("---")
st.caption("🤖 Discount Engine: Because math is cheaper than bargaining.")

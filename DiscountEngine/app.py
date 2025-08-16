import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# -------------------------------------------------
# 1. App Config
# -------------------------------------------------
st.set_page_config(
    page_title="Discount Engine",
    page_icon="💸",
    layout="centered"
)

st.title("💸 Smart Discount Engine")
st.write("Because predicting discounts is cheaper than bargaining 😉")

# -------------------------------------------------
# 2. Load Model (from Hugging Face Hub)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="TanmayPandey9584/Discount_Engine",  # <-- replace with your repo
        filename="discount_model.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# -------------------------------------------------
# 3. User Inputs
# -------------------------------------------------
st.subheader("🔧 Enter Product & Context Details")

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
    sub_category = st.text_input("Sub-Category", "Chairs")
    sales = st.number_input("Sales ($)", min_value=0.0, step=10.0)
    city = st.text_input("City", "Indore")
    state = st.text_input("State", "Madhya Pradesh")

with col2:
    segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
    ship_mode = st.selectbox("Ship Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
    order_month = st.slider("Order Month", 1, 12, 6)
    order_day_of_week = st.slider("Order Day of Week", 0, 6, 2)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])

temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
condition = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Snowy"])

# -------------------------------------------------
# 4. Make Prediction
# -------------------------------------------------
if st.button("💡 Get Discount Prediction"):
    X_pred = pd.DataFrame({
        'Category': [category],
        'Sub-Category': [sub_category],
        'Product ID': ["TEMP123"],  # placeholder since Product ID might not matter
        'Sales': [sales],
        'City': [city],
        'State': [state],
        'Segment': [segment],
        'Ship Mode': [ship_mode],
        'order_month': [order_month],
        'order_day_of_week': [order_day_of_week],
        'season': [season],
        'temperature': [temperature],
        'humidity': [humidity],
        'condition': [condition]
    })

    try:
        discount = model.predict(X_pred)[0]
        st.success(f"🎯 Recommended Discount: **{discount:.2f}%**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------------------------
# 5. Footer (static signature)
# -------------------------------------------------
st.markdown("---")
st.caption("🤖 Discount Engine: Because math is cheaper than bargaining.")



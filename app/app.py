import streamlit as st
import pandas as pd
import json
from pycaret.regression import load_model, predict_model
from pathlib import Path

st.set_page_config(page_title="پیش‌بینی قیمت مسکن در تهران", layout="centered")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;700&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif;
}

body {
    direction: rtl;
}

.stSelectbox div[data-baseweb="select"] {
    direction: rtl;
}

.stSlider {
    direction: ltr;
}

</style>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "tehran_house_price_pipeline"
ADDRESS_PATH = BASE_DIR / "data" / "address_mapping.json"

@st.cache_resource
def load_app_data():
    model = load_model(str(MODEL_PATH))
    with open(ADDRESS_PATH, encoding='utf-8') as f:
        address_map = json.load(f)
    return model, address_map

model, address_mapping = load_app_data()


address_list = [key for key in address_mapping.keys() if key != 'UNKNOWN_DEFAULT']

st.title("پیش‌بینی هوشمند قیمت مسکن در تهران")


col1, col2 = st.columns(2)

with col1:
    area = st.number_input("متراژ (متر مربع)", min_value=30, max_value=2000, value=100, step=5)
    room = st.slider("تعداد اتاق", min_value=0, max_value=5, value=2)
    address = st.selectbox("محله (تایپ یا انتخاب کنید)", address_list)

with col2:
    parking = st.selectbox("پارکینگ", ["دارد", "ندارد"])
    warehouse = st.selectbox("انباری", ["دارد", "ندارد"])
    elevator = st.selectbox("آسانسور", ["دارد", "ندارد"])

if st.button("🔍 پیش‌بینی قیمت"):

    has_parking = True if parking == "دارد" else False
    has_warehouse = True if warehouse == "دارد" else False
    has_elevator = True if elevator == "دارد" else False
    

    encoded_address = address_mapping.get(address, address_mapping['UNKNOWN_DEFAULT'])
    

    input_data = pd.DataFrame({
        'Area': [area],
        'Room': [room],
        'Parking': [has_parking],
        'Warehouse': [has_warehouse],
        'Elevator': [has_elevator],
        'Address_Encoded': [encoded_address]
    }) 
    

    prediction = predict_model(model, data=input_data)
    predicted_price = prediction['prediction_label'][0]
    
    st.success(f"قیمت تخمینی: {predicted_price:,.0f} تومان")

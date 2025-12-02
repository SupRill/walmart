import streamlit as st
import pandas as pd
import pickle

# 1. Judul dan Deskripsi Aplikasi
st.set_page_config(page_title="Walmart Demand Predictor", layout="centered")
st.title("🛒 Walmart Demand Prediction App")
st.write("Aplikasi ini membantu manajer toko memprediksi permintaan barang harian.")

# 2. Load Model
@st.cache_resource
def load_model():
    with open('walmart_demand_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()
model = data['model']
le_weather = data['le_weather']

# 3. Form Input User (Sidebar)
st.sidebar.header("Input Parameter")

unit_price = st.sidebar.number_input("Harga Satuan ($)", min_value=0.0, value=100.0)
inventory_level = st.sidebar.number_input("Level Inventaris Saat Ini", min_value=0, value=50)
lead_time = st.sidebar.slider("Waktu Tunggu Supplier (Hari)", 1, 30, 5)

# Input Kategori (Dropdown)
weather_options = le_weather.classes_
weather = st.sidebar.selectbox("Kondisi Cuaca", weather_options)

# Input Boolean (Checkbox)
is_holiday = st.sidebar.checkbox("Apakah Hari Libur?")
is_promo = st.sidebar.checkbox("Apakah Ada Promosi?")

# 4. Prediksi
if st.button("Prediksi Permintaan"):
    # Preprocessing Input
    weather_encoded = le_weather.transform([weather])[0]
    holiday_encoded = 1 if is_holiday else 0
    promo_encoded = 1 if is_promo else 0
    
    # Buat DataFrame untuk input
    input_data = pd.DataFrame([[unit_price, inventory_level, lead_time, 
                                weather_encoded, holiday_encoded, promo_encoded]],
                              columns=['unit_price', 'inventory_level', 'supplier_lead_time', 
                                       'weather_conditions', 'holiday_indicator', 'promotion_applied'])
    
    # Prediksi
    prediction = model.predict(input_data)
    
    # Tampilkan Hasil
    st.success(f"📊 Prediksi Permintaan Barang: **{int(prediction[0])} unit**")
    
    # Insight Tambahan
    if prediction[0] > inventory_level:
        st.error("⚠️ Peringatan: Potensi STOCKOUT! Permintaan melebihi stok saat ini.")
    else:
        st.info("✅ Stok aman. Tidak perlu re-order mendesak.")

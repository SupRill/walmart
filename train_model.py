import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
df = pd.read_csv('Walmart.csv')

# 2. Fitur Selection
# Kita pilih fitur yang logis mempengaruhi permintaan
features = ['unit_price', 'inventory_level', 'supplier_lead_time', 
            'weather_conditions', 'holiday_indicator', 'promotion_applied']
target = 'actual_demand'

X = df[features].copy()
y = df[target]

# 3. Preprocessing Sederhana
# Encode 'weather_conditions' (Categorical to Number)
le_weather = LabelEncoder()
X['weather_conditions'] = le_weather.fit_transform(X['weather_conditions'])

# Encode Boolean (True/False to 1/0)
X['holiday_indicator'] = X['holiday_indicator'].astype(int)
X['promotion_applied'] = X['promotion_applied'].astype(int)

# 4. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save Model & Encoder
# Kita simpan model dan encoder cuaca agar bisa dipakai di aplikasi
with open('walmart_demand_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'le_weather': le_weather}, f)

print("Model berhasil dilatih dan disimpan sebagai 'walmart_demand_model.pkl'")

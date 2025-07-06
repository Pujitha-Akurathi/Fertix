import pickle
import numpy as np
import pandas as pd
import warnings

# Suppress scikit-learn feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Load Models ---------------- #
FN_model = pickle.load(open("FN_model.pkl", "rb"))
FQ_model = pickle.load(open("FQ_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# ---------------- Fertilizer Class Mapping ---------------- #
fertilizer_dict = {
    1: 'Urea',
    2: 'DAP',
    3: '14-35-14',
    4: '28-28',
    5: '20-20'
}

# ---------------- Fallback Values for Unseen Categories ---------------- #
fallback_values = {
    'soil_type': 'Loamy',
    'crop_type': 'Wheat',
    'crop_stage': 'Harvest',
    'season': 'Summer'
}

# ---------------- Input Sample (Match Training Column Names) ---------------- #
input_data = {
    'temparature': 30.0,              # <== MATCHES model training name
    'humidity': 65.0,
    'moisture': 15.0,
    'soil_type': 'Loamy',             # categorical
    'crop_type': 'Wheat',             # categorical
    'nitrogen': 50.0,
    'crop_stage': 'Harvest',          # categorical
    'acres': 2.5,
    'pH': 6.8,                        # <== MATCHES model training name
    'organic_matter': 3.0,
    'rainfall': 100.0,
    'season': 'Summer',               # categorical
    'potassium': 30.0,
    'phosphorous': 20.0
}

# ---------------- Encode Categorical Fields ---------------- #
for field in ['soil_type', 'crop_type', 'crop_stage', 'season']:
    le = label_encoders[field]
    value = input_data[field]
    try:
        input_data[field] = le.transform([value])[0]
    except ValueError:
        fallback = fallback_values[field]
        print(f"⚠ Warning: '{value}' not found in '{field}'. Using fallback '{fallback}'")
        input_data[field] = le.transform([fallback])[0]

# ---------------- Prepare Input ---------------- #
feature_order = [
    'temparature', 'humidity', 'moisture', 'soil_type', 'crop_type',
    'nitrogen', 'crop_stage', 'acres', 'pH', 'organic_matter',
    'rainfall', 'season', 'potassium', 'phosphorous'
]

features = np.array([[input_data[col] for col in feature_order]])
features_df = pd.DataFrame(features, columns=feature_order)

# ---------------- Make Predictions ---------------- #
fertilizer_class = FN_model.predict(features_df)[0]
fertilizer_quantity = FQ_model.predict(features_df)[0]

fertilizer_name = fertilizer_dict.get(fertilizer_class, "Unknown")

# ---------------- Output ---------------- #
print(f"\n✅ Predicted Fertilizer Name: {fertilizer_name}")
print(f"📦 Predicted Fertilizer Quantity: {fertilizer_quantity:.2f} kg\n")

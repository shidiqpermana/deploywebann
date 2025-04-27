import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import joblib
import os

# Load dan bersihkan nama kolom
df = pd.read_excel('Gastric.xlsx')
df.columns = df.columns.str.strip()

print("Kolom yang tersedia:", df.columns.tolist())

# Sesuaikan dengan nama kolom yang ada
FEATURES = ['diana_microt', 'elmmo', 'microcosm', 'miranda', 'mirdb', 'pictar', 'pita', 'targetscan']
TARGET   = 'all.sum'   # misal ini nama kolom target di file

# Cek ketersediaan kolom
for c in FEATURES + [TARGET]:
    if c not in df.columns:
        raise RuntimeError(f"Kolom '{c}' tidak ditemukan di dataset!")

X = df[FEATURES].values
y = df[TARGET].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Input(shape=(len(FEATURES),)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear'),
])
model.compile(optimizer=Adam(1e-3), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=8,
          validation_data=(X_test, y_test), verbose=1)

# Simpan
os.makedirs('model', exist_ok=True)
model.save('model/ann_model.h5')
joblib.dump(scaler, 'model/scaler.pkl')

print("✅ Training selesai, model & scaler tersimpan.")

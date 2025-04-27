from tensorflow.keras.models import load_model
import joblib

def load_model_and_scaler():
    model = load_model('model/ann_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

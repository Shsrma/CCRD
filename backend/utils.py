import os
import pickle
import numpy as np
from app.core.config import get_settings


# ----------------------------------------
# Load ML Model + Scaler (Safe + Absolute Path)
# ----------------------------------------
def load_model():
    settings = get_settings()
    model_path = settings.model_path
    scaler_path = settings.scaler_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ----------------------------------------
# Preprocess Input Feature Vector
# ----------------------------------------
def preprocess_input(data, scaler):
    """
    Constructs the feature vector in correct order:
    [time, amount, feature_1, feature_2, ..., feature_n]
    """

    # Convert to NumPy array to avoid shape bugs
    X = np.array([data.time, data.amount] + data.features, dtype=float)

    # Reshape to (1, n) for sklearn
    X = X.reshape(1, -1)

    # Scale features and return 1D vector
    return scaler.transform(X)[0]


# ----------------------------------------
# ML Prediction Helper
# ----------------------------------------
def predict_fraud(model, scaler, time, amount, features):
    """
    Make a fraud prediction.
    
    Args:
        model: Trained ML model
        scaler: Fitted scaler
        time: Time of transaction
        amount: Amount of transaction
        features: List of additional features
        
    Returns:
        Tuple of (prediction, probability)
    """
    # Preprocess input
    X = np.array([time, amount] + features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]  # Probability of fraud
    
    return int(prediction), float(probability)
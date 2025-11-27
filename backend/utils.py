import os
import pickle
import numpy as np


# ----------------------------------------
# Load ML Model + Scaler (Safe + Absolute Path)
# ----------------------------------------
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))   # backend/
    model_path = os.path.join(base_dir, "ml", "model.pkl")
    scaler_path = os.path.join(base_dir, "ml", "scaler.pkl")

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

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# -----------------------------
# 1. Resolve Correct Paths
# -----------------------------

# Get absolute path to /ml directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to dataset (adjusted to your project root)
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "creditcard.csv")

# Output paths
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Ensure dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

if "Class" not in df.columns:
    raise ValueError("Dataset is missing required 'Class' column!")

X = df.drop("Class", axis=1)
y = df["Class"]

# -----------------------------
# 3. Scale Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,   # Ensures reproducible model
    stratify=y         # Class-balanced split
)

# -----------------------------
# 5. Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"   # Helps with fraud-data imbalance
)

model.fit(X_train, y_train)

# -----------------------------
# 6. Save Model and Scaler
# -----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("Training Completed!")
print(f"Model saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")

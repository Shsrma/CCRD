import pickle

def load_model():
    model = pickle.load(open("ml/model.pkl", "rb"))
    scaler = pickle.load(open("ml/scaler.pkl", "rb"))
    return model, scaler

def preprocess_input(data, scaler):
    X = [data.time, data.amount] + data.features
    return scaler.transform([X])[0]

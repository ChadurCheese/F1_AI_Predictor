import joblib
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "models/race_predictor.pkl"

def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

import json
import joblib
import pandas as pd
from pathlib import Path
from .data_processing import FEATURES

ROOT = Path(__file__).resolve().parents[1]

def load_model():
    return joblib.load(ROOT / "models" / "trained_model.joblib")

def load_metrics():
    with open(ROOT / "models" / "model_metrics.json", encoding="utf-8") as f:
        return json.load(f)

def predict_one(model, values):
    row = pd.DataFrame([{name: values[name] for name in FEATURES}], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.named_steps["classifier"].classes_)
        if 1 in classes:
            probability = float(model.predict_proba(row)[0][classes.index(1)])
    return prediction, probability

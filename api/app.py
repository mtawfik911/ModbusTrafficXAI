import os
from typing import Literal, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel, Field

from modbus_traffic_xai.preprocessing import load_and_preprocess_data
from modbus_traffic_xai.xai.lime_explainer import explain_one_sample_lime
from modbus_traffic_xai.xai.shap_explainer import explain_with_shap_kernel


# ---- Config ----
INPUT_LENGTH = 17
DATA_PATH = os.getenv("DATA_PATH", "dataset/modbus_traffic_data.csv")

MODEL_PATHS = {
    "cnn": os.getenv("CNN_MODEL_PATH", "models/cnn.keras"),
    "attention": os.getenv("ATT_MODEL_PATH", "models/attention.keras"),
}


# ---- Load background data once (for XAI) ----
X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

# Small background for SHAP speed
SHAP_BACKGROUND = X_train[:100]


def load_model(model_name: str):
    path = MODEL_PATHS[model_name]
    return tf.keras.models.load_model(path)


# Cache models in memory
MODELS = {
    "cnn": load_model("cnn"),
    "attention": load_model("attention"),
}


# ---- API ----
app = FastAPI(title="ModbusTrafficXAI API", version="1.0")


class PredictRequest(BaseModel):
    model: Literal["cnn", "attention"] = "cnn"
    features: list[float] = Field(..., description="17 numeric features (bytes)")


class ExplainRequest(BaseModel):
    model: Literal["cnn", "attention"] = "cnn"
    features: list[float] = Field(..., description="17 numeric features (bytes)")
    lime_top_k: int = 10
    shap_top_k: int = 10
    shap_nsamples: int = 200


@app.get("/health")
def health():
    return {
        "status": "ok",
        "input_length": INPUT_LENGTH,
        "models": MODEL_PATHS,
        "train_samples": int(X_train.shape[0]),
    }


def _predict_prob(model, features_1d: np.ndarray) -> float:
    x = features_1d.astype(np.float32).reshape(1, INPUT_LENGTH, 1)
    prob = float(model.predict(x, verbose=0)[0][0])
    return prob


@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.features, dtype=np.float32)

    if x.shape[0] != INPUT_LENGTH:
        return {"error": f"features length must be {INPUT_LENGTH}, got {x.shape[0]}"}

    model = MODELS[req.model]
    prob = _predict_prob(model, x)
    label = 1 if prob >= 0.5 else 0

    return {"model": req.model, "label": label, "probability": prob}


@app.post("/explain")
def explain(req: ExplainRequest):
    x = np.array(req.features, dtype=np.float32)

    if x.shape[0] != INPUT_LENGTH:
        return {"error": f"features length must be {INPUT_LENGTH}, got {x.shape[0]}"}

    model = MODELS[req.model]

    # ---- LIME ----
    lime_exp = explain_one_sample_lime(
        model=model,
        X_train=X_train,
        sample=x,
        num_features=req.lime_top_k,
    )
    lime_list = [{"rule": r, "weight": float(w)} for r, w in lime_exp.as_list()]

    # ---- SHAP ----
    # Explain one sample (shape (1, features))
    sample_2d = x.reshape(1, INPUT_LENGTH)
    _, shap_values = explain_with_shap_kernel(
        model=model,
        X_background=SHAP_BACKGROUND,
        X_samples=sample_2d,
        nsamples=req.shap_nsamples,
    )

    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    v = np.squeeze(shap_vals[0])  # (features,)
    abs_vals = np.abs(v)
    top_idx = np.argsort(abs_vals)[::-1][: req.shap_top_k]

    feature_names = [f"byte_{i}" for i in range(INPUT_LENGTH)]
    shap_top = [
        {"feature": feature_names[int(i)], "shap_value": float(v[int(i)])}
        for i in top_idx
    ]

    prob = _predict_prob(model, x)
    label = 1 if prob >= 0.5 else 0

    return {
        "model": req.model,
        "label": label,
        "probability": prob,
        "lime_top": lime_list,
        "shap_top": shap_top,
    }


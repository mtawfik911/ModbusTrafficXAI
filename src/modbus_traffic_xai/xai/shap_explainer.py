import numpy as np
import shap


def _model_predict(model):
    """
    SHAP will call this with X shape (n, features).
    We reshape to (n, features, 1) for Conv1D models and return probabilities.
    """
    def predict_fn(X):
        X = np.asarray(X, dtype=np.float32)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return model.predict(X, verbose=0)
    return predict_fn


def explain_with_shap_kernel(model, X_background, X_samples, nsamples=200):
    """
    KernelExplainer (model-agnostic, slower but robust).
    - X_background: (m, features) background data (use small m like 50-200)
    - X_samples: (n, features) samples to explain
    Returns: shap_values (list or array depending on model output)
    """
    X_background = np.asarray(X_background, dtype=np.float32)
    X_samples = np.asarray(X_samples, dtype=np.float32)

    explainer = shap.KernelExplainer(_model_predict(model), X_background)
    shap_values = explainer.shap_values(X_samples, nsamples=nsamples)
    return explainer, shap_values


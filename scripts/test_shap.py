import numpy as np
import tensorflow as tf
import shap
import os
import matplotlib.pyplot as plt

from modbus_traffic_xai.preprocessing import load_and_preprocess_data
from modbus_traffic_xai.xai.shap_explainer import explain_with_shap_kernel

DATA_PATH = "dataset/modbus_traffic_data.csv"

# Choose model:
MODEL_PATH = "models/cnn.keras"
# MODEL_PATH = "models/attention.keras"


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Background: small subset for speed
    background = X_train[:100]

    # Explain 1 sample
    sample = X_test[:1]  # shape (1, features)

    explainer, shap_values = explain_with_shap_kernel(
        model=model,
        X_background=background,
        X_samples=sample,
        nsamples=200
    )

    # KernelExplainer may return list for multi-output models
    if isinstance(shap_values, list):
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    feature_names = [f"byte_{i}" for i in range(X_train.shape[1])]

    print("✅ SHAP values computed.")
    print("Top SHAP contributions (absolute):")

    # ---- robust shaping: take first sample and squeeze to 1D (features,)
    v = shap_vals[0]
    v = np.squeeze(v)  # handles (features,), (features,1), (features,1,1), etc.

    # Ensure 1D
    if v.ndim != 1:
        raise ValueError(f"Unexpected SHAP vector shape after squeeze: {v.shape}")

    abs_vals = np.abs(v)
    top_idx = np.argsort(abs_vals)[::-1][:10]

    for i in top_idx:
        i = int(i)
        print(f"{feature_names[i]:10s}  shap={v[i]:+.6f}")

    # ---- optional plot (may require GUI support)
    # Create a (1, features) array for summary_plot
    v2d = np.array([v])
    os.makedirs("results/shap", exist_ok=True)

    shap.summary_plot(
        v2d,
        sample,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()
    plt.savefig("results/shap/shap_summary_sample0.png", dpi=300)
    plt.close()

    print("✅ Saved plot: results/shap/shap_summary_sample0.png")

if __name__ == "__main__":
    main()
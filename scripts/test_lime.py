import tensorflow as tf
import os
from modbus_traffic_xai.preprocessing import load_and_preprocess_data
from modbus_traffic_xai.xai.lime_explainer import explain_one_sample_lime

DATA_PATH = "dataset/modbus_traffic_data.csv"

# Choose model
MODEL_PATH = "models/cnn.keras"
# MODEL_PATH = "models/attention.keras"

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)

    idx = 0
    sample = X_test[idx]

    exp = explain_one_sample_lime(model, X_train, sample, num_features=10)

    print("✅ LIME explanation (top features):")
    for rule, weight in exp.as_list():
        print(f"{rule:30s}  weight={weight:+.4f}")

    os.makedirs("results/lime", exist_ok=True)
    exp.save_to_file("results/lime/lime_sample0.html")
    print("✅ Saved LIME HTML: results/lime/lime_sample0.html")

if __name__ == "__main__":
    main()



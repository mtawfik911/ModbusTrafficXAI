import os
import numpy as np
import tensorflow as tf

from modbus_traffic_xai.preprocessing import load_and_preprocess_data
from modbus_traffic_xai.models import build_cnn


DATA_PATH = "dataset/modbus_traffic_data.csv"
MODEL_OUT = "models/cnn.keras"


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    # Add channel dim for Conv1D: (N, 17) -> (N, 17, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    model = build_cnn(input_length=X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_OUT)
    print(f"✅ Saved model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
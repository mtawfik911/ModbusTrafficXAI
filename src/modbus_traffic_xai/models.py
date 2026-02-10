import tensorflow as tf
from tensorflow.keras import layers


def build_cnn(input_length: int = 17) -> tf.keras.Model:
    """
    CNN model (no-attention) for Modbus packet classification.
    Input: (input_length, 1)
    Output: sigmoid probability (normal vs abnormal)
    """
    inputs = tf.keras.Input(shape=(input_length, 1), name="packet_bytes")

    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_no_attention")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model




def build_attention_cnn(input_length: int = 17) -> tf.keras.Model:
    """
    Attention + CNN model for Modbus packet classification.
    Uses MultiHeadAttention + residual connection + LayerNorm (standard practice).
    """
    inputs = tf.keras.Input(shape=(input_length, 1), name="packet_bytes")

    # Project to higher dimension so attention is meaningful
    x = layers.Dense(32, activation=None)(inputs)  # (19, 32)

    # Self-attention
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
    x = layers.Add()([x, attn])           # residual
    x = layers.LayerNormalization()(x)    # stabilize

    # CNN stack
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)

    model = tf.keras.Model(inputs, outputs, name="attention_cnn")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


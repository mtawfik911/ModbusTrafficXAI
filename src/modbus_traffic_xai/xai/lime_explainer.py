import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def _keras_predict_proba(model):
    """
    LIME expects predict_fn(X) -> (n_samples, n_classes)
    We return 2 classes: [P(normal=0), P(abnormal=1)].
    """

    def predict_fn(X):
        X = np.asarray(X, dtype=np.float32)

        # reshape to (n, features, 1) for Conv1D models
        X = X.reshape((X.shape[0], X.shape[1], 1))

        p1 = model.predict(X, verbose=0).reshape(-1)   # P(class=1)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    return predict_fn


def explain_one_sample_lime(
    model,
    X_train,
    sample,
    feature_names=None,
    class_names=("normal", "abnormal"),
    num_features=10,
):
    """
    Explain one sample with LIME.
    - X_train: (n, 17) background data
    - sample: (17,)
    Returns: LIME explanation object
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    sample = np.asarray(sample, dtype=np.float32)

    if feature_names is None:
        feature_names = [f"byte_{i}" for i in range(X_train.shape[1])]

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
    )

    exp = explainer.explain_instance(
        data_row=sample,
        predict_fn=_keras_predict_proba(model),
        num_features=min(num_features, X_train.shape[1]),
    )
    return exp


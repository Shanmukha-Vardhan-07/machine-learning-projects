"""Load the saved model and make predictions."""
import joblib
import pandas as pd
import sklearn

from .config import FEATURES, MODEL_PATH


def load_model(auto_train: bool = True):
    """Load the model. If it is missing, or was saved with a different
    scikit-learn version, retrain it so we never unpickle an incompatible file."""
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        if bundle.get("sklearn_version") == sklearn.__version__:
            return bundle["model"]
        print("Saved model was built with a different scikit-learn version; retraining...")
    elif not auto_train:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python -m src.train")
    from .train import train
    return train(verbose=False)


def predict(model, sepal_length, sepal_width, petal_length, petal_width):
    """Return (species, confidence_percent_or_None)."""
    X = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                     columns=FEATURES)
    species = model.predict(X)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = round(float(proba[list(model.classes_).index(species)]) * 100, 1)
    return species, confidence


if __name__ == "__main__":
    m = load_model()
    print(predict(m, 5.1, 3.5, 1.4, 0.2))

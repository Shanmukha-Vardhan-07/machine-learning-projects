"""Train and compare models, then save the deployed one.

Run from the project root:  python -m src.train
"""
import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import (COMPARISON_CSV, FEATURES, MODEL_PATH, RANDOM_STATE,
                     TARGET)
from .preprocessing import get_clean_data, split

DEPLOYED_MODEL = "KNN"  # the web app is built around KNN


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }


def train(verbose: bool = True):
    df = get_clean_data(save=True)
    X_train, X_test, y_train, y_test = split(df)

    rows, fitted = [], {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        cv = cross_val_score(model, df[FEATURES], df[TARGET], cv=5)
        rows.append({"Model": name,
                     "Test Accuracy": round(test_acc, 4),
                     "CV Mean Accuracy": round(cv.mean(), 4),
                     "CV Std": round(cv.std(), 4)})
        fitted[name] = model
        if verbose:
            print(f"{name:20s} test={test_acc:.3f}  cv={cv.mean():.3f}")

    results = (pd.DataFrame(rows)
               .sort_values("CV Mean Accuracy", ascending=False))
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(COMPARISON_CSV, index=False)

    # Save the model together with the sklearn version that built it, so the
    # app can detect a version mismatch and retrain instead of crashing.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[DEPLOYED_MODEL],
                 "sklearn_version": sklearn.__version__}, MODEL_PATH)
    if verbose:
        print(f"\nSaved {DEPLOYED_MODEL} -> {MODEL_PATH}")
        print(f"Saved comparison -> {COMPARISON_CSV}")
    return fitted[DEPLOYED_MODEL]


if __name__ == "__main__":
    train()

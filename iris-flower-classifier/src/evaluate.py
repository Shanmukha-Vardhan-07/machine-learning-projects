"""Evaluate the saved model and write figures to reports/figures.

Run from the project root:  python -m src.evaluate
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)

from .config import COMPARISON_CSV, FIGURES_DIR, TARGET
from .predict import load_model
from .preprocessing import get_clean_data, split


def evaluate():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = get_clean_data(save=False)
    model = load_model()
    _, X_test, _, y_test = split(df)
    y_pred = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Purples",
                                            xticks_rotation=30)
    ax.set_title("Confusion matrix (KNN)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    if COMPARISON_CSV.exists():
        comp = pd.read_csv(COMPARISON_CSV)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=comp, x="Model", y="CV Mean Accuracy", ax=ax, color="#7c5cbf")
        ax.set_ylim(0.8, 1.0)
        ax.set_title("Model comparison (5-fold CV)")
        plt.setp(ax.get_xticklabels(), rotation=20)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
        plt.close(fig)

    sns.pairplot(df, hue=TARGET).savefig(FIGURES_DIR / "pairplot.png", dpi=110)
    plt.close("all")
    print("Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    evaluate()

"""Flask web app. Run from the project root:  python app/app.py"""
import sys
from pathlib import Path

# Make the `src` package importable no matter where the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, render_template, request  # noqa: E402

from src.predict import load_model, predict  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__,
            template_folder=str(APP_DIR / "templates"),
            static_folder=str(APP_DIR / "static"))

model = load_model()  # auto-retrains if the file is missing / version-mismatched

DEFAULTS = {"sepal_length": 5.8, "sepal_width": 3.0,
            "petal_length": 3.8, "petal_width": 1.2}

SPECIES_INFO = {
    "Iris-setosa": {"common_name": "Setosa",
                    "fact": "Smallest of the three — short, rounded petals and a compact flower."},
    "Iris-versicolor": {"common_name": "Versicolor",
                        "fact": "The in-betweener — medium-sized petals and sepals."},
    "Iris-virginica": {"common_name": "Virginica",
                       "fact": "The largest — long petals and sepals set it apart from the rest."},
}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, values=DEFAULTS)


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        values = {k: float(request.form[k]) for k in DEFAULTS}
    except (KeyError, ValueError):
        return render_template(
            "index.html", values=DEFAULTS,
            result={"error": "Please enter valid numbers for all four measurements."})

    species, confidence = predict(model, **values)
    info = SPECIES_INFO.get(species, {"common_name": species, "fact": ""})
    result = {"species": species, "common_name": info["common_name"],
              "fact": info["fact"], "confidence": confidence}
    return render_template("index.html", result=result, values=values)


if __name__ == "__main__":
    app.run(debug=True)

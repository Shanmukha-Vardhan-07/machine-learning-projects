"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw" / "Iris.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "iris_clean.csv"
MODEL_PATH = ROOT / "models" / "iris_model.pkl"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"
RANDOM_STATE = 42
TEST_SIZE = 0.2

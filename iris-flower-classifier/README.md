# Iris Flower Classification

An end-to-end machine learning project that classifies iris flowers into **Setosa, Versicolor, or Virginica** based on four flower measurements.

## Overview

This project compares five classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

After comparing the models using cross-validation and evaluation metrics, the best-performing model is selected and deployed using Flask.

## Features

- Data preprocessing and validation
- Exploratory Data Analysis (EDA)
- Comparison of 5 machine learning models
- 5-fold cross-validation
- Accuracy, precision, recall and F1-score evaluation
- Confusion matrix
- Model comparison
- Model serialization using Joblib
- Flask web application for predictions

## Project Structure

```text
Iris-Flower-Classification/
├── data/
│   ├── raw/
│   │   └── Iris.csv
│   └── processed/
├── notebooks/
│   └── iris_classification.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/
│   └── iris_model.pkl
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
├── reports/
│   ├── figures/
│   └── model_comparison.csv
├── requirements.txt
└── README.md
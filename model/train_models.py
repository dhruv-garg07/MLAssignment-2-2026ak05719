"""
BITS Pilani WILP - M.Tech (AIML)
Machine Learning - Assignment 2
Training & Evaluation Script for 5 Classification Models

Dataset: UCI Iranian Telecom Customer Churn Dataset (ID: 563)
Instances: 3,150 | Input Features: 13 | Target: Churn (0 = Non-Churn, 1 = Churn)
"""

from __future__ import annotations

import io
import json
import platform
import urllib.request
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 2026
TARGET_COLUMN = "Churn"
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
DATASET_PATH = MODEL_DIR / "dataset.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"


def load_or_download_dataset() -> pd.DataFrame:
    """Load the dataset from disk, or download it from UCI if missing."""
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
    else:
        print(f"Downloading UCI Iranian Churn dataset from {UCI_DATASET_URL} ...")
        # Hardcoded official UCI HTTPS endpoint (no user-controlled URL input)
        raw_bytes = urllib.request.urlopen(UCI_DATASET_URL, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
            if not csv_files:
                raise RuntimeError("No CSV file found inside UCI dataset archive.")
            df = pd.read_csv(archive.open(csv_files[0]))
        df.columns = [" ".join(col.strip().split()) for col in df.columns]
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)

    df.columns = [" ".join(col.strip().split()) for col in df.columns]
    return df


def build_models() -> dict[str, tuple[str, Pipeline]]:
    """Define the 5 required classification pipelines with feature scaling."""
    return {
        "Logistic Regression": (
            "logistic_regression.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(C=1.2, max_iter=1000, random_state=RANDOM_SEED)),
                ]
            ),
        ),
        "Decision Tree": (
            "decision_tree.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        DecisionTreeClassifier(
                            max_depth=7,
                            min_samples_leaf=4,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        ),
        "KNN": (
            "knn.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
                ]
            ),
        ),
        "Naive Bayes": (
            "naive_bayes.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", GaussianNB()),
                ]
            ),
        ),
        "Support Vector Machine": (
            "svm.joblib",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(C=1.5, kernel="rbf", probability=True, random_state=RANDOM_SEED),
                    ),
                ]
            ),
        ),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calculate all 6 required evaluation metrics."""
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def main() -> None:
    print("=" * 78)
    print("BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2")
    print(f"Environment : Python {platform.python_version()} | scikit-learn {sklearn.__version__} | Host: {platform.node()}")
    print("=" * 78)

    df = load_or_download_dataset()
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found in dataset.")

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    X = df[feature_columns]
    y = df[TARGET_COLUMN].astype(int)

    print(f"Dataset     : UCI Iranian Telecom Customer Churn")
    print(f"Instances   : {len(df)} rows (Requirement >= 500)")
    print(f"Features    : {len(feature_columns)} input features (Requirement >= 12)")
    print(f"Target Dist : Non-Churn (0) = {int((y == 0).sum())}, Churn (1) = {int((y == 1).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"Split       : Train = {len(X_train)} rows | Test = {len(X_test)} rows (20% stratified)")
    print("-" * 78)

    # Save held-out test dataset (features + target column) for Streamlit evaluation
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved held-out test dataset -> {TEST_DATA_PATH.name} ({len(test_df)} rows, {len(test_df.columns)} columns)")
    print("-" * 78)

    models = build_models()
    results_summary: dict[str, dict[str, float]] = {}
    model_files: dict[str, str] = {}

    header = f"{'ML Model Name':<25} {'Accuracy':>9} {'AUC':>9} {'Precision':>10} {'Recall':>9} {'F1':>9} {'MCC':>9}"
    print(header)
    print("-" * 78)

    for model_name, (filename, pipeline) in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)
        results_summary[model_name] = metrics
        model_files[model_name] = filename

        save_path = MODEL_DIR / filename
        joblib.dump(pipeline, save_path)

        print(
            f"{model_name:<25} "
            f"{metrics['Accuracy']:>9.4f} "
            f"{metrics['AUC']:>9.4f} "
            f"{metrics['Precision']:>10.4f} "
            f"{metrics['Recall']:>9.4f} "
            f"{metrics['F1']:>9.4f} "
            f"{metrics['MCC']:>9.4f}"
        )

    metadata = {
        "dataset_name": "UCI Iranian Telecom Customer Churn Dataset",
        "dataset_url": "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        "total_instances": int(len(df)),
        "train_instances": int(len(X_train)),
        "test_instances": int(len(X_test)),
        "num_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "class_labels": {"0": "Non-Churn", "1": "Churn"},
        "sklearn_version": sklearn.__version__,
        "model_files": model_files,
        "benchmark_metrics": results_summary,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("-" * 78)
    best_model = max(results_summary.items(), key=lambda item: (item[1]["F1"], item[1]["MCC"]))
    print(f"Best Performing Model: {best_model[0]} (F1 = {best_model[1]['F1']:.4f}, MCC = {best_model[1]['MCC']:.4f})")
    print(f"Saved all 5 trained pipelines and metadata in: {MODEL_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()

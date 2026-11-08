"""
BITS Pilani WILP - M.Tech (AIML)
Machine Learning - Assignment 2
Interactive Streamlit Application for Telecom Churn Classification Models
"""

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Paths anchored strictly to repository root
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR / "model").resolve()
DEFAULT_TEST_CSV = (BASE_DIR / "test_data.csv").resolve()
METADATA_FILE = (MODEL_DIR / "metadata.json").resolve()

# Strict allow-list of pre-trained model artifacts
MODEL_FILES: dict[str, str] = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB upload cap to prevent DoS
MAX_UPLOAD_ROWS = 20_000

# TODO(security): Public academic Streamlit demo has no user accounts, cookies, or database;
# add authentication/OAuth/MFA and rate limiting if restricting access in an enterprise deployment.
# TODO(security): Uploaded CSVs are parsed strictly in-memory as numeric matrices and never saved to disk;
# integrate sandboxed malware/CDR scanning if persistent file storage is ever introduced.

st.set_page_config(
    page_title="Telecom Churn Classifier | ML Assignment 2",
    layout="wide",
)


@st.cache_data
def load_metadata() -> dict:
    """Load feature schema and dataset details saved during training."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Model metadata.json is missing. Run model/train_models.py first.")
    return json.loads(METADATA_FILE.read_text(encoding="utf-8"))


@st.cache_resource
def load_trained_models() -> dict[str, object]:
    """Load all 5 pre-trained classification pipelines from disk without retraining."""
    loaded: dict[str, object] = {}
    for model_name, filename in MODEL_FILES.items():
        candidate_path = (MODEL_DIR / filename).resolve()
        # Enforce strict directory boundary check
        if candidate_path.parent != MODEL_DIR or not candidate_path.is_file():
            raise FileNotFoundError(f"Saved model artifact not found for {model_name}.")
        loaded[model_name] = joblib.load(candidate_path)
    return loaded


@st.cache_data
def load_default_test_csv() -> tuple[pd.DataFrame, bytes]:
    """Load the bundled held-out test dataset."""
    raw_bytes = DEFAULT_TEST_CSV.read_bytes()
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = [" ".join(str(col).strip().split()) for col in df.columns]
    return df, raw_bytes


def validate_and_parse_csv(uploaded_file, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    """Validate uploaded CSV extension, size, schema, and numeric values."""
    filename = str(getattr(uploaded_file, "name", ""))
    if not filename.lower().endswith(".csv"):
        raise ValueError("Invalid file type. Please upload a valid .csv file.")

    raw_bytes = uploaded_file.getvalue()
    if len(raw_bytes) == 0:
        raise ValueError("The uploaded CSV file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file exceeds the 5 MB size limit.")

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError("Unable to parse CSV file. Ensure it is properly formatted UTF-8 CSV.") from exc

    df.columns = [" ".join(str(col).strip().split()) for col in df.columns]

    if len(df) == 0:
        raise ValueError("The uploaded CSV contains headers but zero data rows.")
    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Uploaded CSV exceeds maximum supported row count ({MAX_UPLOAD_ROWS}).")

    required_cols = feature_cols + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in uploaded CSV: {', '.join(missing_cols)}")

    clean_df = df[required_cols].copy()
    for col in required_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    if clean_df.isna().any().any():
        raise ValueError("Uploaded CSV contains missing or non-numeric values in required columns.")

    unique_targets = set(clean_df[target_col].unique().tolist())
    if not unique_targets.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target column '{target_col}' must contain binary values (0 or 1).")

    clean_df[target_col] = clean_df[target_col].astype(int)
    return clean_df


def evaluate_model(model, X_eval: pd.DataFrame, y_eval: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Run inference using a saved model and compute all 6 evaluation metrics."""
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    if len(np.unique(y_eval)) > 1:
        auc_val = float(roc_auc_score(y_eval, y_prob))
    else:
        auc_val = float("nan")

    metrics = {
        "Accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
        "AUC": round(auc_val, 4) if not np.isnan(auc_val) else float("nan"),
        "Precision": round(float(precision_score(y_eval, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_eval, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_eval, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_eval, y_pred)), 4),
    }
    return metrics, y_pred


def render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Plot a clean 2x2 confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Churn (0)", "Churn (1)"],
        yticklabels=["Non-Churn (0)", "Churn (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    return fig


def main() -> None:
    st.title("Telecom Customer Churn — ML Classification Suite")
    st.caption(
        "BITS Pilani WILP | M.Tech (AIML) — Machine Learning Assignment 2 | "
        "Dataset: UCI Iranian Telecom Customer Churn (3,150 instances, 13 features)"
    )

    try:
        metadata = load_metadata()
        models = load_trained_models()
        default_df, default_csv_bytes = load_default_test_csv()
    except Exception:
        st.error("Application initialization failed: ensure saved model artifacts and test_data.csv exist.")
        return

    feature_cols: list[str] = metadata["feature_columns"]
    target_col: str = metadata["target_column"]

    # Sidebar controls: CSV upload + Model dropdown
    with st.sidebar:
        st.header("Evaluation Controls")

        selected_model_name = st.selectbox(
            "1. Select Classification Model",
            options=list(MODEL_FILES.keys()),
            index=0,
            help="Choose any of the 5 pre-trained classifiers saved in model/.",
        )

        st.divider()
        st.subheader("2. Upload Test Dataset (CSV)")
        uploaded_file = st.file_uploader(
            "Upload test CSV (13 features + Churn column)",
            type=["csv"],
            help="Upload held-out test data only. Models run inference directly without retraining.",
        )

        st.download_button(
            label="Download Sample test_data.csv",
            data=default_csv_bytes,
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )

        st.divider()
        st.caption(
            f"Pre-trained on {metadata['train_instances']} training instances "
            f"({metadata['num_features']} features). No retraining occurs on uploaded data."
        )

    # Determine active test dataset
    if uploaded_file is not None:
        try:
            active_df = validate_and_parse_csv(uploaded_file, feature_cols, target_col)
            data_source_label = f"Uploaded Test CSV ({len(active_df)} rows)"
            st.success(f"Loaded uploaded test CSV successfully ({len(active_df)} rows, {len(feature_cols)} features + target '{target_col}').")
        except ValueError as err:
            st.error(str(err))
            st.info("Falling back to bundled test_data.csv.")
            active_df = default_df
            data_source_label = f"Bundled test_data.csv ({len(active_df)} rows)"
    else:
        active_df = default_df
        data_source_label = f"Bundled test_data.csv ({len(active_df)} rows)"
        st.info(
            f"Evaluating on bundled held-out `{DEFAULT_TEST_CSV.name}` ({len(active_df)} rows). "
            "Use the sidebar uploader to test any custom CSV slice."
        )

    with st.expander(f"Preview Active Test Data — {data_source_label}", expanded=False):
        st.dataframe(active_df.head(15), width="stretch")

    X_eval = active_df[feature_cols]
    y_eval = active_df[target_col].to_numpy(dtype=int)

    # Evaluate selected model
    selected_pipeline = models[selected_model_name]
    selected_metrics, selected_preds = evaluate_model(selected_pipeline, X_eval, y_eval)

    st.subheader(f"Evaluation Metrics — {selected_model_name}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
    m2.metric("AUC Score", f"{selected_metrics['AUC']:.4f}" if not np.isnan(selected_metrics["AUC"]) else "N/A")
    m3.metric("Precision", f"{selected_metrics['Precision']:.4f}")
    m4.metric("Recall", f"{selected_metrics['Recall']:.4f}")
    m5.metric("F1 Score", f"{selected_metrics['F1']:.4f}")
    m6.metric("MCC Score", f"{selected_metrics['MCC']:.4f}")

    st.divider()

    # Confusion Matrix & Classification Report side-by-side
    col_cm, col_cr = st.columns([1, 1.2])

    with col_cm:
        st.subheader("Confusion Matrix")
        fig = render_confusion_matrix(y_eval, selected_preds, selected_model_name)
        st.pyplot(fig, width="content")
        plt.close(fig)

    with col_cr:
        st.subheader("Classification Report")
        report_dict = classification_report(
            y_eval,
            selected_preds,
            labels=[0, 1],
            target_names=["Non-Churn (0)", "Churn (1)"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        st.dataframe(report_df, width="stretch")

    st.divider()

    # Comparison of all 5 models on active test data
    st.subheader("All 5 Models — Performance Comparison on Test Data")
    comparison_rows = []
    for name, pipeline in models.items():
        m_vals, _ = evaluate_model(pipeline, X_eval, y_eval)
        comparison_rows.append({"ML Model Name": name, **m_vals})

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}",
            }
        ).highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            props="background-color: #15803d; color: #ffffff; font-weight: bold;",
        ),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()

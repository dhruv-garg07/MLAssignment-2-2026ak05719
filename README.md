# Telecom Customer Churn Prediction — ML Classification Suite

**BITS Pilani Work Integrated Learning Programmes (WILP)**  
**M.Tech (AIML) — Machine Learning | Assignment 2**

---

## a. Problem Statement

Customer churn (attrition) is a critical revenue challenge in the telecommunications industry, as acquiring new subscribers costs substantially more than retaining existing ones. The objective of this project is to build, evaluate, and deploy an end-to-end supervised binary classification pipeline that predicts whether a telecom subscriber will churn (`1`) or remain active (`0`) based on call quality, billing behavior, service complaints, and usage patterns. Five machine learning classification algorithms are trained on a common stratified split, compared rigorously across six evaluation metrics, and served via an interactive Streamlit web application for real-time test-set inference.

---

## b. Dataset Description

- **Dataset Name:** UCI Iranian Churn Dataset
- **Public Repository Source:** [UCI Machine Learning Repository — Iranian Churn Dataset (ID: 563)](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
- **Problem Type:** Binary Classification (`0` = Non-Churn, `1` = Churn)
- **Total Instances:** **3,150** rows *(exceeds minimum requirement of 500 instances)*
- **Input Feature Count:** **13** numerical features *(exceeds minimum requirement of 12 features)*
- **Missing Values:** **0** (complete dataset with no null entries)
- **Class Distribution:**
  - `Non-Churn (0)`: 2,655 instances (84.29%)
  - `Churn (1)`: 495 instances (15.71%)
- **Train / Test Partition:** 80% Training (`2,520` rows) / 20% Stratified Held-Out Test (`630` rows saved in `test_data.csv`), `random_state=2026`. Each classifier pipeline applies `StandardScaler` fitted strictly on the training split.

### Input Features (13 Attributes)

| # | Feature Name | Description |
| :---: | :--- | :--- |
| 1 | `Call Failure` | Number of call failures experienced by the subscriber |
| 2 | `Complains` | Binary indicator (`0` = No complaint, `1` = Complaint filed) |
| 3 | `Subscription Length` | Total months of active subscription |
| 4 | `Charge Amount` | Ordinal billing tier (`0` lowest to `9` highest) |
| 5 | `Seconds of Use` | Total call duration in seconds |
| 6 | `Frequency of use` | Total number of calls placed |
| 7 | `Frequency of SMS` | Total number of text messages sent |
| 8 | `Distinct Called Numbers` | Count of unique phone numbers contacted |
| 9 | `Age Group` | Ordinal age bracket (`1` youngest to `5` oldest) |
| 10 | `Tariff Plan` | Plan type (`1` = Pay-as-you-go, `2` = Contractual) |
| 11 | `Status` | Account activity status (`1` = Active, `2` = Non-active) |
| 12 | `Age` | Subscriber age in years |
| 13 | `Customer Value` | Calculated lifetime monetary value of the subscriber |

- **Target Column:** `Churn` (`0` = Non-Churn, `1` = Churn)

---

## c. Github Repository Link

- **GitHub Repository Link:** [https://github.com/dhruv-garg07/MLAssignment-2-2026ak05719](https://github.com/dhruv-garg07/MLAssignment-2-2026ak05719)
- **Live Streamlit App Link:** [https://mlassignment-2-2026ak05719-elxhrtsfbkpbpc8dczlcrg.streamlit.app/](https://mlassignment-2-2026ak05719-elxhrtsfbkpbpc8dczlcrg.streamlit.app/)

### Repository Structure

```text
MLAssignment-2-2026ak05719/
├── app.py                         # Interactive Streamlit web application (inference-only)
├── requirements.txt               # Python package dependencies for deployment
├── README.md                      # Project documentation, metrics, and model analysis
├── test_data.csv                  # Held-out test split (630 rows: 13 features + Churn target)
└── model/
    ├── dataset.csv                # Full UCI Iranian Churn dataset (3,150 rows)
    ├── train_models.py            # Training & evaluation source script
    ├── metadata.json              # Feature schema and benchmark metadata
    ├── logistic_regression.joblib # Saved Logistic Regression pipeline
    ├── decision_tree.joblib       # Saved Decision Tree pipeline
    ├── knn.joblib                 # Saved K-Nearest Neighbor pipeline
    ├── naive_bayes.joblib         # Saved Gaussian Naive Bayes pipeline
    └── svm.joblib                 # Saved Support Vector Machine pipeline
```

---

## d. Models Used

### Comparison Table of Evaluation Metrics (Evaluated on Held-Out `test_data.csv`)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8810 | 0.9269 | 0.7308 | 0.3838 | 0.5033 | 0.4728 |
| **Decision Tree** | 0.9286 | 0.9307 | 0.8000 | 0.7273 | 0.7619 | 0.7211 |
| **KNN** | 0.9492 | 0.9674 | 0.8681 | 0.7980 | 0.8316 | 0.8027 |
| **Naive Bayes** | 0.7460 | 0.9056 | 0.3755 | 0.9293 | 0.5349 | 0.4786 |
| **Support Vector Machine** | 0.9238 | 0.9653 | 0.9636 | 0.5354 | 0.6883 | 0.6854 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Serves as a solid linear baseline (`Accuracy = 0.8810`, `AUC = 0.9269`), showing good probability ranking ability. However, because churners represent 15.71% of subscribers and exhibit non-linear interactions across complaints and usage drop-offs, its linear decision boundary is overly conservative—yielding low `Recall (0.3838)`, `F1 (0.5033)`, and `MCC (0.4728)`. |
| **Decision Tree** | Captures non-linear threshold splits across key drivers (`Complains`, `Status`, `Seconds of Use`, and `Subscription Length`), raising `Recall` sharply from `0.3838` to `0.7273` while maintaining `Precision = 0.8000`. Depth regularization (`max_depth=7`) controls variance and produces strong balanced performance (`F1 = 0.7619`, `MCC = 0.7211`). |
| **KNN** | Achieves the strongest overall performance across nearly all metrics (`Accuracy = 0.9492`, `AUC = 0.9674`, `F1 = 0.8316`, `MCC = 0.8027`). With `StandardScaler` normalization and distance-weighted `k=7` voting, subscribers with similar call-failure, complaint, and activity profiles cluster tightly in the 13-dimensional feature space, balancing high `Precision (0.8681)` and `Recall (0.7980)`. |
| **Naive Bayes** | Delivers the highest `Recall (0.9293)` among all five models (identifying ~93% of true churners), making it aggressive at flagging at-risk customers. However, strong collinearity among telecom usage variables (`Seconds of Use`, `Frequency of use`, `Charge Amount`, and `Customer Value`) violates Gaussian Naive Bayes' conditional independence assumption, causing excessive false positives (`Precision = 0.3755`, `Accuracy = 0.7460`). |
| **Support Vector Machine** | Using an RBF kernel, SVM achieves the highest `Precision (0.9636)` of all models alongside near-top `AUC (0.9653)`, meaning almost every predicted churner is genuinely churning. Under the 84:16 class imbalance, the default `0.5` probability threshold favors margin certainty on the majority class, resulting in moderate `Recall (0.5354)` and `MCC (0.6854)`. |
| **Overall, Winner for your dataset?** | **KNN (K-Nearest Neighbor Classifier)** is the clear overall winner on this dataset. Because the target classes are imbalanced (`84.29%` Non-Churn vs. `15.71%` Churn), **F1 Score (`0.8316`)** and **Matthews Correlation Coefficient (`MCC = 0.8027`)** are the most reliable selection criteria, and KNN leads all five classifiers in Accuracy (`94.92%`), AUC (`0.9674`), F1 (`0.8316`), and MCC (`0.8027`). |

---

## Local Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-train models and regenerate test_data.csv
python model/train_models.py

# 3. Launch the Streamlit application locally
streamlit run app.py --server.address 127.0.0.1
```

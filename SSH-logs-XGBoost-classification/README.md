# SSH Log Attack Classification using eXtreme Gradient Boosting (XGBoost)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Framework: XGBoost](https://img.shields.io/badge/Framework-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

An end-to-end Machine Learning pipeline designed to parse, transform, and classify SSH log signatures into malicious brute-force/unauthorized access vectors (`Class 1`) or legitimate session profiles (`Class 0`).

---

## 1. Introduction

Securing Secure Shell (SSH) access points is a critical mandate for modern infrastructure management. Brute-force discovery attacks, unauthorized user harvesting, and scanning credentials pollute network footprints. 

This project implements an intelligent intrusion detection system (IDS) utilizing behavioral telemetry captured within structured raw SSH session elements. By extracting temporal behaviors, normalizing heavy distributional right-skewness, and deploying an eXtreme Gradient Boosting (`XGBClassifier`) engine, this framework yields near-faultless security anomaly detection capabilities, maintaining zero missed threat indicators (`100% recall`).

---

## 2. Requirements & Environment Setup

The workspace is configured inside a localized `Python 3.12` or Kaggle Docker standard container environment.

### Core Framework Dependencies
- **Data Arrays & Structs**: `numpy`, `pandas`
- **Modeling & Pipeline Mechanics**: `scikit-learn`
- **Gradient Boost Engine**: `xgboost`
- **Analytical Graphics Engine**: `matplotlib`, `seaborn`

To replicate the installation matrix, run:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn

```

---

## 3. Dataset Characteristics

* **Ingestion Reference Path**: `/kaggle/input/datasets/osamac/ssh-logs-with-attack-classification/SSH.csv`
* **Shape Matrix**: 283 records $\times$ 13 initial structural columns.
* **Null / Missing Value Contingency**: 0 instances across all records (clean dataset out-of-the-box).

### Target Vector Distribution (`class`)

The distribution reveals an imbalanced real-world threat profile:

* **Class 0 (Normal / Legitimate Access)**: 216 records ($76.3\%$)
* **Class 1 (Malicious Intrusions / Attacks)**: 67 records ($23.7\%$)

---

## 4. Preprocessing & Feature Engineering Operations

Raw structural entries are subjected to a rigorous engineering chain to conform numerical distributions and resolve high-cardinality categorical profiles:

### A. Temporal Feature Dissection

The numeric Unix-based timestamp variable (`ts`) is expanded to capture human behavioral patterns through local chronologies. It generates four new discrete columns:

* `hour` (0–23)
* `dayofweek` (0–6)
* `day` (1–31)
* `month` (1–12)

The root attributes `ts` and the temporary timestamp object `dt` are scrubbed to eliminate multicollinearity overhead.

### B. Distributional Skewness Stabilization

Numerical tracking metrics reveal extreme right-skew patterns due to massive outliers during malicious multi-try execution loops. To minimize variance shifts, a natural log transformer ($\ln(x + 1)$) via `np.log1p` is sequentially applied to:

* `not_valid_count`
* `ip_failure`
* `ip_success`
* `td` (Time Duration)
* `no_failure`

### C. Mean Target Likelihood Value Mapping

The user category field contains broad unique textual string names (e.g., `osamac`, `kamran`, `root`). To prevent sparse dimensionality spikes caused by One-Hot-Encoding, **Mean Target Encoding** is implemented:

$$\text{Encoded Value} = P(\text{Class} = 1 \mid \text{User})$$

This explicitly maps categorical labels to their calculated historical threat probability distributions.

---

## 5. Model Architecture & Pipeline Setup

The data matrix is split using a stratified **80/20 train/test partition** to enforce exact cross-strata representation:

* **Training Dataset Subspace**: 226 records (15 engineered features)
* **Testing Evaluation Subspace**: 57 records (15 engineered features)

### Model Engine Selection: XGBoost

An optimized `XGBClassifier` is designated due to its superior capacity to process tabular class imbalances via gradient tree-boosting routines.

```python
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42,
    colsample_bytree=0.8,
    subsample=0.8
)

```

---

## 6. Performance & Evaluation Metrics

### A. Phase Accuracies

* **Training Phase Accuracy**: **100.00%** (Perfect alignment over learned operational signatures)
* **Testing / Validation Accuracy**: **98.24%**

### B. Test Set Classification Report

Below is the model performance evaluation over unseen target records:

| Class Indicator | Operational Meaning | Precision | Recall (Sensitivity) | F1-Score | Support Base |
| --- | --- | --- | --- | --- | --- |
| **Class 0** | Normal Session / Secure Entry | 1.00 | 0.98 | 0.99 | 44 |
| **Class 1** | Malicious Attack Signature | 0.93 | 1.00 | 0.96 | 13 |
| **Macro Avg** | Global Category Average | 0.96 | 0.99 | 0.97 | 57 |
| **Weighted Avg** | Volume-Weighted Evaluation | 0.98 | 0.98 | 0.98 | 57 |

### C. Confusion Matrix Breakdown

The absolute count matrices outline flawless malicious detection over test elements:

```text
                      Predicted Legitimate (0)   Predicted Attack (1)
True Legitimate (0)             43                          1
True Attack (1)                  0                         13

```

* **True Negatives (TN)**: **43** legitimate connections correctly validated.
* **False Positives (FP)**: **1** clean connection incorrectly flagged as a security threat.
* **False Negatives (FN)**: **0** missed anomalies.
* **True Positives (TP)**: **13** malicious access signatures successfully isolated.

---

## 7. Operational Highlights & Observations

1. **Zero False Negative Security Operations**: The system records a **1.00 (100%) Recall** score for Class 1 threats. In real-world enterprise infrastructure, letting a single attack slip through undetected is catastrophic, making high recall a core performance win.
2. **Minimal False-Alarm Overhead**: Out of all valid system usages, only one instance triggered an anomalous alarm. This minimizes system alert fatigue for Security Operations Center (SOC) engineers.
3. **Engineering Impact**: Dissecting temporal features and smoothing numeric outliers with logarithmic functions allowed the gradient boosting trees to easily establish clear decision boundaries despite the small size of the dataset.

---

## 8. Conclusion

This project demonstrates that compiling sparse textual system files into clean, feature-rich telemetry matrices enables highly effective security classification models. By executing log transforms alongside categorical target likelihood assignments, our `XGBoost` model yields an impressive **98.24% validation accuracy** and captures **100% of all real attack threats**. Future versions will look into integrating streaming infrastructures like Kafka to handle real-time SSH parsing and automated IP blocking.

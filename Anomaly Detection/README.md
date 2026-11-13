# Anomaly Detection Engineering Portfolio

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Portfolio-Job%20Ready-111827)

## Executive Summary

This repository demonstrates an end-to-end anomaly detection workflow using 2 complementary approaches:

1. **Isolation Forest** — fast, tree-based outlier isolation for high-dimensional transaction data.
2. **Gaussian Mixture Models + Density Estimation** — probabilistic modeling of multimodal normal behavior.


The implementation is intentionally modular. Shared evaluation and visualization utilities live in `src/anomaly_detection/`, while each notebook can be run independently after the relevant dataset is downloaded.


## Problem Statement

Rare-event detection appears in fraud prevention, cybersecurity, predictive maintenance, infrastructure monitoring, quality control, and healthcare signal monitoring. The central challenge is that the positive class is often tiny, labels may be incomplete, and the operating cost of false positives can be substantial.



## Dataset Strategy

The datasets are intentionally matched to the modeling stage rather than forcing every algorithm onto one benchmark.

| Model stage | Ideal data characteristics | Recommended dataset | Reference |
|---|---|---|---|
| Isolation Forest | High-dimensional tabular transactions; severe class imbalance | Credit Card Fraud Detection | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| GMM + Density | Continuous network/sensor variables with multimodal normal regimes | Network Intrusion Detection | [Kaggle](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection) |




See [`data/README.md`](data/README.md) for dataset placement and reproducibility guidance.

## Repository Structure

```text
anomaly_detection_portfolio/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── README.md
│   ├── Isolation_forest.ipynb
│   └── Gmm_density.ipynb
├── src/
│   └── anomaly_detection/
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
└── reports/
    ├── README.md
    └── figures/
```

## Data Science Lifecycle

### 1. Business Understanding

**Business question:** Which observations should be escalated as suspicious or abnormal while keeping false-positive workload manageable?

Examples:

- Fraud: prioritize high-risk transactions for manual review.
- Network security: prioritize suspicious flows for SOC investigation.
- Healthcare signals: surface ECG patterns that merit specialist review.
- Infrastructure: detect behavior that deviates from learned operating regimes.

### 2. Data Acquisition

Download the relevant Kaggle dataset, document the dataset version/date, and place the extracted CSV under `data/`.

Never commit private or licensed source data to GitHub unless the license explicitly permits redistribution.

### 3. Data Quality & EDA

The notebooks implement a defensively engineered preprocessing layer:

- Numeric feature selection.
- Infinite-value replacement.
- Median imputation for missing numeric values.
- Robust or standard scaling depending on algorithm and variable behavior.
- Explicit class-prevalence reporting.
- Train/test stratification when labels are available.
- Normal-only training for novelty-detection workflows.

### 4. Feature Engineering

Feature engineering is deliberately conservative in this baseline. The objective is to isolate algorithmic behavior rather than hide it behind a large feature-engineering pipeline.

For a production extension, consider:

- Transaction velocity features.
- Rolling network statistics.
- Session-level aggregates.
- Signal-window normalization.
- Domain-specific frequency-domain features for ECG.

### 5. Modeling Strategy

#### Isolation Forest

Isolation Forest randomly partitions observations. Samples requiring shorter isolation paths are more likely to be anomalous. The notebook exposes `n_estimators`, `max_samples`, and contamination assumptions. Current scikit-learn documentation defines `contamination` as the expected outlier proportion used to establish the decision threshold. 

#### Gaussian Mixture Models

GMM models a continuous distribution as a mixture of Gaussian components. Component count and covariance structure are selected using BIC, with AIC reported as a secondary criterion. Scikit-learn exposes both `aic(X)` and `bic(X)`, with lower values preferred; its model-selection example uses BIC to choose among component counts and covariance types. 


### 6. Threshold Selection

Threshold selection is treated as an explicit modeling decision rather than a hidden library default.

| Model | Primary score | Threshold logic |
|---|---|---|
| Isolation Forest | `-score_samples(X)` | Library contamination threshold; validation PR-AUC used for hyperparameter selection |
| GMM | `-log_density` | 1st percentile of normal training log-density |

For deployment, threshold selection should be tied to a business constraint such as maximum review volume, minimum recall, or target precision.

### 7. Evaluation

Accuracy is intentionally excluded from the primary evaluation story because a model that labels almost every transaction as normal can appear highly accurate in a heavily imbalanced dataset.

Primary metrics:

- **Precision:** how many flagged observations are truly anomalous.
- **Recall:** how many known anomalies are captured.
- **F1:** harmonic balance between precision and recall.
- **PR-AUC / Average Precision:** primary ranking metric under severe imbalance.
- **ROC-AUC:** secondary ranking metric.
- **Inference time:** operational suitability.



## LinkedIn Portfolio Showcase

Recommended visual assets to export from `reports/figures/`:

| Asset | Recruiter message |
|---|---|
| Isolation Forest decision landscape | “I can turn high-dimensional anomaly scores into interpretable operational visuals.” |
| GMM density contours | “I can model multimodal normal behavior rather than treating every deviation as a simple distance.” |



## Reproducibility

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
jupyter lab
```






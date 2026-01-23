# Temporal Drift & Uncertainty Analysis in Deep Learning-Based Glucose Forecasting

## Overview
This project studies how temporal drift affects prediction accuracy, uncertainty calibration, and explainability in deep learning-based glucose forecasting using Continuous Glucose Monitoring (CGM) data.

An LSTM-based deep learning model is trained on early time segments and evaluated on future chronological windows to simulate real-world deployment scenarios where data distributions evolve over time.

---

## Key Objectives
- Perform time-aware (chronological) evaluation instead of random train-test splits
- Analyze performance degradation caused by temporal drift
- Estimate epistemic uncertainty using Monte Carlo Dropout
- Study explainability drift using SHAP feature attributions

---

## Dataset
- **OhioT1DM Dataset (Kaggle)**
- Raw dataset is **not included** in this repository due to licensing restrictions
- Users must download the dataset separately from Kaggle and provide the local CSV path

---

## Methodology
- Feature engineering on CGM time-series data
- LSTM-based sequence modeling for glucose forecasting
- Monte Carlo Dropout for uncertainty estimation
- SHAP for model explainability analysis
- Temporal segmentation into four chronological windows (T1–T4)

---

## Results Summary
- Model performance degrades over later temporal windows due to distribution shift
- Prediction intervals become under-calibrated under temporal drift
- Feature importance patterns shift significantly over time

---

## Repository Structure

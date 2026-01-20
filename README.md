# Temporal Drift & Uncertainty Analysis in Deep Learning-Based Glucose Forecasting

## Overview
This project studies how temporal drift affects prediction accuracy, uncertainty calibration, and explainability in deep learning-based glucose forecasting using Continuous Glucose Monitoring (CGM) data.

An LSTM model is trained on early time segments and evaluated on future chronological windows to simulate real-world deployment scenarios.

## Key Objectives
- Perform time-aware (chronological) evaluation instead of random train-test splits
- Analyze performance degradation due to temporal drift
- Estimate epistemic uncertainty using Monte Carlo Dropout
- Study explainability drift using SHAP feature attributions

## Dataset
- OhioT1DM Dataset (Kaggle)
- Raw data is not included due to licensing restrictions

## Methodology
- Feature engineering on CGM time series
- LSTM-based sequence modeling
- Monte Carlo Dropout for uncertainty estimation
- SHAP for explainability analysis
- Temporal segmentation into four windows (T1–T4)

## Results Summary
- Model performance degrades over later temporal windows
- Prediction intervals are under-calibrated under drift
- Feature importance shifts significantly over time

## Repository Structure

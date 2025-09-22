# Gamma Ray Classification Using Machine Learning

High-performance binary classification system for distinguishing gamma ray signals from hadron background noise in telescope data.

## Project Overview

Built a production-ready classification pipeline achieving **93.8% ROC-AUC** for particle physics data, showcasing skills directly transferable to quantitative finance:
- Signal detection in noisy data
- Feature engineering from complex measurements
- Model selection and hyperparameter optimization
- Risk-adjusted performance metrics

## Key Results

- **Accuracy**: 88.7%
- **ROC-AUC**: 93.8%
- **Precision**: 89.0%
- **Recall**: 94.3%

## Technical Highlights

### Data Processing
- Handled 19,020 observations with 10 continuous features
- Implemented robust standardization and outlier detection
- Addressed class imbalance (65/35 split)

### Model Development
- Evaluated 5 different algorithms with systematic hyperparameter tuning
- Gradient Boosting emerged as optimal (200 estimators, max_depth=7)
- Comprehensive cross-validation strategy (5-fold stratified)

### Feature Analysis
- Identified key discriminative features through importance analysis
- fAlpha (angle) and size parameters most predictive
- Aligned findings with domain physics understanding

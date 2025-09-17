# MAGIC Gamma Telescope Classification

End-to-end machine-learning pipeline to classify **gamma rays vs. hadron particles** from telescope measurements.  
Built for the UCI MAGIC Gamma Telescope dataset (~19k samples, 10 features).  
Final model: **Gradient Boosting**, achieving **~0.887 accuracy and 0.938 ROC-AUC**.

---

## 🚀 Highlights
- **Full ML pipeline**: data cleaning, feature standardization, EDA, dimensionality reduction (PCA/t-SNE), model selection, and validation.
- **Modeling & tuning**:
  - Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting.
  - Grid/Randomized cross-validated hyperparameter search.
- **Best model**: Gradient Boosting (200 trees, max depth 7, learning rate 0.1, subsample 0.8).
- **Scientific insight**: Feature importance matches physics expectations  
  (`fAlpha`, `fLength`, `fSize` most discriminative).

---

## 🗂 Repository Structure
magic-gamma-classification/
├─ README.md
├─ requirements.txt # python packages (scikit-learn, pandas, matplotlib, etc.)
├─ notebooks/
│ ├─ 01_eda.ipynb # exploratory analysis & visualizations
│ ├─ 02_baselines.ipynb # logistic regression & SVM baselines
│ ├─ 03_model_sweeps.ipynb
│ ├─ 04_feature_importance.ipynb
│ └─ 05_report_figures.ipynb
├─ src/
│ ├─ data.py # load/clean/standardize, stratified train/test split
│ ├─ models.py # build and configure models
│ ├─ train.py # cross-validation, metrics, saving results
│ ├─ plots.py # ROC, confusion matrix, feature importances
│ └─ utils.py # seeds, logging, helpers
├─ configs/ # YAML experiment configs
│ └─ gb_best.yaml
├─ results/
│ ├─ metrics.csv # summary of CV/test results
│ └─ figures/ # saved ROC curves, confusion matrices, etc.
└─ data/
└─ README.md # instructions to download raw data

🔑 Key Results
Model	Accuracy	ROC-AUC	F1
Logistic Regression	0.78	0.83	0.84
SVM (RBF kernel)	0.87	0.93	0.91
Random Forest	0.88	0.94	0.91
Gradient Boosting	0.89	0.94	0.92

Representative figures are saved under results/figures/:

roc_curve.png

confusion_matrix.png

feature_importance.png

🧩 Physics & Feature Insights

fAlpha (orientation angle) is the dominant feature (~28 % importance), consistent with gamma rays arriving near the telescope’s center.

fLength and fSize capture shower shape and energy, key to separating gamma from hadron events.

This alignment between ML importance and astrophysical theory provides scientific validation.

Contact

Created by Yonathan Amare
M.A. Statistics, Columbia University
ya2562@columbia.edu

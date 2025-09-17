MAGIC Gamma Telescope Classification

End-to-end machine-learning pipeline to classify gamma rays vs. hadron particles from telescope measurements
using the UCI MAGIC Gamma Telescope dataset
 (~19k samples, 10 features).
Final model: Gradient Boosting, achieving ~0.89 accuracy and 0.94 ROC-AUC.

Highlights

Full ML workflow: data loading & cleaning, feature standardization, exploratory analysis, PCA & t-SNE visualization.

Modeling: Logistic Regression, K-Nearest Neighbors, SVM, Random Forest, and Gradient Boosting.

Hyperparameter tuning: Grid/Randomized search with stratified 5-fold cross-validation.

Best model: Gradient Boosting (n_estimators=200, max_depth=7, learning_rate=0.1, subsample=0.8).

Physics insight: Top features (fAlpha, fLength, fSize) align with astrophysical theory.

Repository Structure
notebooks/         # EDA and model exploration
src/               # Reusable Python modules (data prep, training, plotting)
configs/           # YAML configs for reproducible experiments
results/           # Saved metrics, figures, and final models (if desired)
data/              # Place raw data here (not committed); see instructions below
requirements.txt   # Python dependencies
README.md

Quickstart
git clone https://github.com/<your-username>/magic-gamma-classification.git
cd magic-gamma-classification
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download dataset from UCI and place it in data/
python scripts/run_experiment.py --config configs/gb_best.yaml


Key figures (ROC curve, confusion matrix, feature importance) are stored in results/figures/.

Key Results
Model	Accuracy	ROC-AUC	F1 Score
Logistic Regression	0.78	0.83	0.84
SVM (RBF)	0.87	0.93	0.91
Random Forest	0.88	0.94	0.91
Gradient Boosting	0.89	0.94	0.92

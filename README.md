# BOI Hackathon - Mule Account Detection

This repository contains a mule account detection system built for the BOI hackathon. The project uses a machine learning pipeline to detect suspicious accounts from a transactional dataset.

## Project Summary

- **Dataset size:** 9,082 accounts
- **Suspicious accounts identified:** 81 (0.89%)
- **Best model:** XGBoost
- **Final performance:** Precision 1.0000, Recall 1.0000, F1 1.0000, AUC-ROC 1.0000
- **Selected features after preprocessing:** 120

## Files

- `main.py` - Main pipeline script for preprocessing, feature selection, model training, evaluation, and visualization.
- `DataSet.csv` - Input dataset used for training and evaluation.
- `risk_profile.csv` - Generated risk score output for test accounts.
- `feature_importance.csv` - Ranked model feature importances.
- `complete_dashboard.png` - Generated dashboard with model and analysis visualizations.
- `feature_importance_plot.png` - Feature importance chart.
- `mi_distribution.png` - Mutual information distribution plots.
- `submission_summary.txt` - Summary of results and generated outputs.
- `requirements.txt` - Required Python packages.

## Pipeline Overview

The system follows these major steps:

1. Load and clean the dataset.
2. Convert feature values to numeric and drop invalid columns.
3. Remove features with extreme missing values or near-zero variance.
4. Impute missing values using the median.
5. Select top features using mutual information.
6. Remove highly correlated features.
7. Scale features using `RobustScaler`.
8. Balance the training set using SMOTE-ENN.
9. Train XGBoost and Random Forest models.
10. Evaluate models using precision, recall, F1 score, and AUC.
11. Generate visualizations and output risk scores.

## Usage

1. Create a Python environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the main script.

```bash
python main.py
```

3. Review generated outputs in the repository root.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- catboost
- imbalanced-learn

## Notes

- `main.py` expects `DataSet.csv` to be present in the same directory.
- The model pipeline includes data cleaning, mutual information feature selection, correlation filtering, and ensemble training.
- The best model reported in `submission_summary.txt` is XGBoost with perfect precision and recall on the evaluation split.

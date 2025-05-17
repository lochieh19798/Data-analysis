# data-analysis
# Recurrence\_2yr Prediction with Machine Learning

## Overview

This repository demonstrates a complete machine-learning pipeline for predicting 2-year recurrence of atrial fibrillation post-cryoablation using clinical and procedural data. The workflow includes:

* **Data loading & cleaning**
* **Missing-value analysis**
* **Preprocessing pipelines** (numeric KNN imputation + scaling, categorical encoding + imputation)
* **Train/test splitting** (random 80/20 and temporal hold-out)
* **Model training & tuning** across multiple algorithms:

  * Logistic Regression
  * Support Vector Machine (SVM)
  * Naive Bayes
  * ExtraTrees
  * AdaBoost
  * HistGradientBoosting
  * XGBoost (native CV for rounds + randomized search)
  * CatBoost (randomized search)
* **Evaluation**:

  * ROC curves (ShuffleSplit mean ± std)
  * Calibration curve & Brier score
  * Bootstrap AUC 95% CI
* **Explainability**:

  * SHAP analyses for XGBoost and CatBoost
  * Partial dependence plots
  * (Optional) Decision curve analysis

## Repository Structure

```
├── CRYOANALYSIS.csv           # Raw clinical dataset (upload separately)
├── notebook.ipynb             # Main analysis notebook
└── README.md                  # This file
```

## Requirements

* Python 3.10+
* pandas
* numpy
* scikit-learn==1.4.2
* xgboost>=1.7
* catboost
* shap
* matplotlib

Install dependencies via:

```bash
pip install -r requirements.txt
```

*(Alternatively)* in Colab:

```bash
pip install -q catboost shap scikit-learn==1.4.2 matplotlib pandas numpy
pip install -q --upgrade xgboost
```

## Usage

1. **Clone** the repository and place `CRYOANALYSIS.csv` in the root directory.
2. **(Optional)** Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. **Launch** Jupyter Notebook or Lab:

   ```bash
   jupyter lab
   ```
5. **Open** `notebook.ipynb` and run cells in order:

   * **Cell 0**: install missing packages (Colab only)
   * **Cell 1**: imports & global config
   * **Cell 2**: load & prepare data
   * **Cell 3**: train/test split (random and temporal)
   * **Cell 4a–4d**: preprocessing pipelines & feature inspection
   * **Cell 4.5–4.9**: hyperparameter tuning for XGBoost, ExtraTrees, HistGB, CatBoost, AdaBoost
   * **Cell 5**: assemble final models dictionary
   * **Cell 6**: evaluate ROC (ShuffleSplit)
   * **Cell 7b**: bootstrap AUC 95% CI
   * **Cell 9.5–10**: calibration curves & Brier scores
   * **Cell 10**: SHAP plots for XGBoost & CatBoost
   * **Cell 14**: partial dependence plots
   * **Cell 16**: decision curve analysis (if library installed)
6. **Inspect** the output plots and metrics. Customize hyperparameters or split strategies as needed.

## Notes & Tips

* **Data Privacy**: Ensure patient identifiers are removed before sharing or versioning.
* **Runtime**: Large hyperparameter searches can be time-consuming—consider reducing `n_iter` or using smaller grids for local experimentation.
* **External Validation**: If you have an independent cohort, replace or augment the temporal hold-out split to assess generalization.
* **Reproducibility**: All random seeds are set via `RANDOM_STATE=42`.

---

Feel free to open issues or contribute improvements!

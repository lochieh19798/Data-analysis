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
├── Figure2mod.ipynb           # Main analysis notebook
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
5. **Open** `Figure2mod.ipynb` and run cells in order:

   * **Cell 0** – install packages (Colab only)
   * **Cell 1** – import libraries and set options
   * **Cell 2** – load the dataset and build feature lists
   * **Cell 3** – create train/test splits
   * **Cell 4** – build the preprocessing pipeline
   * **Cell 4b** – preview numeric imputation
   * **Cell 4c** – preview categorical imputation
   * **Cell 4d** – inspect the preprocessed feature matrix
   * **Cell 4.4** – helper for CatBoost categorical handling
   * **Cell 4.5** – tune XGBoost rounds with native CV
   * **Cell 4.6** – grid search ExtraTrees
   * **Cell 4.8** – random search HistGradientBoosting
   * **Cell 5** – assemble the models dictionary
   * **Cell 5.1** – quick CatBoost search
   * **Cell 5.2** – tune AdaBoost parameters
   * **Cell 5.3** – final XGBoost with early stopping
   * **Cell 6** – evaluate ROC via ShuffleSplit
   * **Cell 7** – plot the combined ROC figure
   * **Cell 7b** – bootstrap AUC 95% CI
   * **Cell 8** – bar chart of mean AUCs
   * **Cell 9** – summary table of AUCs
   * **Cell 9.5** – apply sigmoid calibration
   * **Cell 10** – calibration curve and Brier score
   * **Cell 11a** – precision–recall and threshold tuning
   * **Cell 11b** – SHAP summary for XGBoost
   * **Cell 12** – partial dependence plots
   * **Cell 13** – permutation importance
   * **Cell 14** – decision curve analysis
   * **Cell 15** – SHAP summary for CatBoost
   * **Cell 16** – partial dependence for CatBoost
   * **Cell 17** – permutation importance for CatBoost
   * **Cell 18** – decision curve for CatBoost
6. **Inspect** the output plots and metrics. Customize hyperparameters or split strategies as needed.

## Notes & Tips

* **Data Privacy**: Ensure patient identifiers are removed before sharing or versioning.
* **Runtime**: Large hyperparameter searches can be time-consuming—consider reducing `n_iter` or using smaller grids for local experimentation.
* **External Validation**: If you have an independent cohort, replace or augment the temporal hold-out split to assess generalization.
* **Reproducibility**: All random seeds are set via `RANDOM_STATE=42`.

---

Feel free to open issues or contribute improvements!

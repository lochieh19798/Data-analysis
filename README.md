# data-analysis
# Recurrence\_2yr Prediction with Machine Learning

## Overview

This repository demonstrates a complete machine-learning pipeline for predicting 2-year recurrence of atrial fibrillation post-cryoablation using clinical and procedural data. The workflow includes:

* **Data loading & cleaning**
* **Missing-value analysis**
* **Preprocessing pipelines** (numeric iterative imputation → power transform → scaling; categorical constant imputation → one-hot encoding)
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
pip install catboost shap scikit-learn==1.4.2 matplotlib pandas numpy
pip install --upgrade xgboost
```

*(Alternatively)* in Colab use the same commands with `-q` for quiet output.*

## Usage

1. **Clone** the repository and place `CRYOANALYSIS.csv` in the root directory.
2. **(Optional)** Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install** dependencies:

   ```bash
   pip install catboost shap scikit-learn==1.4.2 matplotlib pandas numpy
   pip install --upgrade xgboost
   ```
4. **Launch** Jupyter Notebook or Lab:

   ```bash
   jupyter lab
   ```
5. **Open** `Figure2mod.ipynb` and run cells in order:

   * **Cell 0** – install packages (Colab only)
   * **Cell 1** – import libraries and set options
   * **Cell 2** – load data & build feature lists with missing-% report
   * **Cell 3** – stratified train/test/validation split
   * **Cell 4** – preprocessing pipelines
   * **Cell 4b** – preview IterativeImputer + PowerTransformer on numeric features
   * **Cell 4c** – preview categorical imputation + one-hot encoding
   * **Cell 4d** – inspect the full preprocessed matrix
   * **Cell 4.4** – helper for CatBoost categorical handling
   * **Cell 4.5** – tune XGBoost rounds with native CV
   * **Cell 4.6** – randomized ExtraTrees search with SMOTE
   * **Cell 4.8** – randomized HistGradientBoosting search with SMOTE
   * **Cell 5** – build models dictionary with SMOTE in pipelines
   * **Cell 5.1** – full-train smoke test of all pipelines
   * **Cell 5.2** – quick CatBoost tuning on SMOTE-resampled data
   * **Cell 5.3** – tune AdaBoost parameters with SMOTE
   * **Cell 5.4** – final XGBoost search with early stopping
   * **Cell 6** – ShuffleSplit ROC (mean ± std)
   * **Cell 7** – plot combined ROC figure
   * **Cell 7b** – bootstrap AUC 95% CI
   * **Cell 8** – bar chart of mean AUCs
   * **Cell 9** – summary table of AUCs
   * **Cell 10.1** – pipeline + sigmoid calibration
   * **Cell 10.2** – calibration curve and Brier score
   * **Cell 10.3** – precision–recall & threshold tuning
   * **Cell 10.4** – SHAP summary (model-agnostic)
   * **Cell 10.5** – partial dependence plots
   * **Cell 10.6** – permutation importance
   * **Cell 10.7** – decision curve analysis
   * **Cell 10.8** – final evaluation on future hold-out
   * **Cell 10.9** – bootstrap-calibrated XGBoost
   * **Cell 11.0** – SHAP summary for CatBoost
   * **Cell 11.1** – partial dependence for CatBoost
   * **Cell 11.2** – permutation importance for CatBoost
   * **Cell 11.3** – decision curve for CatBoost
   * **Cell 11.4** – final evaluation on future hold-out for CatBoost
   * **Cell 11.5** – calibrate CatBoost on the build set
   * **Cell 11.6** – bootstrap-ensemble for CatBoost
   * **Cell 11.7** – compare calibrated vs bootstrap CatBoost
6. **Inspect** the output plots and metrics. Customize hyperparameters or split strategies as needed.

## Notes & Tips

* **Data Privacy**: Ensure patient identifiers are removed before sharing or versioning.
* **Runtime**: Large hyperparameter searches can be time-consuming—consider reducing `n_iter` or using smaller grids for local experimentation.
* **External Validation**: If you have an independent cohort, replace or augment the temporal hold-out split to assess generalization.
* **Reproducibility**: All random seeds are set via `RANDOM_STATE=42`.

---

Feel free to open issues or contribute improvements!

# Data Analysis

## Recurrence_2yr Prediction with Machine Learning

This repository contains a Jupyter notebook demonstrating a full machine-learning pipeline for predicting 2‑year atrial fibrillation recurrence after cryoablation. The main analysis is in **`Figure2mod.ipynb`**.

### Repository Contents

```
├── CRYOANALYSIS.csv  # Raw dataset (not versioned)
├── Figure2mod.ipynb  # Main analysis notebook
├── Figure2mod        # Binary output from notebook
└── README.md         # Project documentation
```

### Getting Started

1. Clone the repository.
2. Place `CRYOANALYSIS.csv` in the project root.
3. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
5. Launch Jupyter Lab and open `Figure2mod.ipynb`:

   ```bash
   jupyter lab
   ```
6. Run the notebook cells in order to reproduce the analysis.

### Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn==1.4.2
- xgboost>=1.7
- catboost
- shap
- matplotlib

### Notes

- Ensure all patient identifiers are removed before sharing data.
- Large hyperparameter searches can be slow; adjust the search space for faster experiments.
- Random seeds are set via `RANDOM_STATE=42` for reproducibility.

---

Feel free to open issues or contribute improvements!

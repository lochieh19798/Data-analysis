# data_utils.py
import pandas as pd
from pathlib import Path

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the CRYOANALYSIS CSV, does basic cleaning, and
    creates Recurrence_1yr/2yr flags and AF parox/persist columns.
    """
    df = pd.read_csv(csv_path)
    # Example transformations from your notebook:
    df["Survival_time"] = pd.to_numeric(df["Survival_time"], errors="coerce")
    df = df.dropna(subset=["Survival_time", "Recurrence"])
    df["Recurrence_1yr"] = ((df["Survival_time"] <= 365) & (df["Recurrence"] == 1)).astype(int)
    df["Recurrence_2yr"] = ((df["Survival_time"] <= 730) & (df["Recurrence"] == 1)).astype(int)

    af_col = "Baseline AF Type(1=paroxysmal, 2=persistent)"
    df["AF_Parox"]   = (df[af_col] == 1).astype(int)
    df["AF_Persist"] = (df[af_col] == 2).astype(int)

    return df


# Your continuous & categorical feature lists:
cont = [
    "age", "BMI",
    "Baseline LVEF", "Baseline LAD", "CHA2DS2VASc score", "CHAD2 score",
    "AF_time_procedure",
    "Total no of ablation application number",
    "Mobility question (Baseline)",
    "Self-care question (Baseline)",
    "Usual activities question (Baseline)",
    "Pain/Discomfort question (Baseline)",
    "Anxiety/Depression question (Baseline)",
    "Visual analogue score: Your own health state today (Baseline)",
    "Total procedure time: Venous access to last cryoatheter removal (mins)",
    "Total fluoro time (mins)",
    "Energy duration LSPV", "Coldest Temperature LSPV", "Time to isolation LSPV",
    "Energy duration LIPV", "Coldest Temperature LIPV", "Time to isolation LIPV",
    "Energy duration RSPV", "Coldest temperature RSPV", "Time to isolation RSPV",
    "Energy duration RIPV", "Coldest Temperature RIPV", "Time to isolation RIPV",
    "Left atrial dwell time: time from first cryocatheter insertion to last cryocatheter removal (mins)",
    "Change in EQ 5D"
]

cat = [
    "Sex (F=1, M=0)", "Hypertension", "Diabetes", "HF", "CAD", "stroke",
    "History of TIA", "Subject taking Class I or III AAD at baseline (1=Yes, 0=No)",
    "LSPV isolated", "LIPV PV isolated", "RSPV Isolated", "RIPV isolated",
    "Were all targeted PVs isolated (Investigator)?",
    "AF_Parox", "AF_Persist",
    "CTI ablation",
    "Non-PVI ablation performed",
    "Was subject taking Class I or Class III AAD at procedure discharge?",
    "Mapping/navigational tools: Intracardiac echocardiography (ICE)",
    "Pre Procedural CT",
    "Ensite 3D mapping"
]

TARGET = "Recurrence_2yr"

def prepare_for_catboost(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """
    Copy X, replace any NaN in the specified categorical columns
    with a literal 'Missing' string (so CatBoost sees it as a valid category).
    Leave numeric columns as-is (CatBoost can handle numeric NaNs).
    """
    X_cb = X.copy()
    for c in categorical_cols:
        if c in X_cb.columns:
            X_cb[c] = X_cb[c].fillna("Missing").astype(str)
    return X_cb

# modeling.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def build_model_dict(
    preprocessor,
    best_et_params=None,
    best_hgb_params=None,
    best_ab_params=None,
    best_xgb_params=None,
    catboost_estimator=None
):
    """
    Return a dictionary of model name → model object.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        A fitted or unfitted preprocessor pipeline (e.g., KNNImputer + OneHotEncoder).
        Typically built from your 'build_preprocessor()' function.
    best_et_params : dict, optional
        Best ExtraTrees hyperparameters, e.g.:
        {
          "n_estimators": 300,
          "max_depth": 10,
          "max_features": "sqrt",
          "min_samples_split": 5,
          "min_samples_leaf": 2
        }
    best_hgb_params : dict, optional
        Best HistGradientBoostingClassifier hyperparameters, e.g.:
        {
          "learning_rate": 0.03,
          "max_iter": 1000,
          "max_depth": 5,
          "min_samples_leaf": 50,
          "l2_regularization": 0.1,
          "max_leaf_nodes": 31
        }
    best_ab_params : dict, optional
        Best AdaBoost hyperparameters, e.g.:
        {
          "n_estimators": 300,
          "learning_rate": 0.5
        }
    best_xgb_params : dict, optional
        Best XGBoost hyperparameters, e.g.:
        {
          "n_estimators": 500,
          "learning_rate": 0.01,
          "max_depth": 5,
          "subsample": 0.8,
          "colsample_bytree": 0.8,
          "gamma": 1.0,
          "reg_alpha": 0.1,
          "reg_lambda": 1.0
        }
    catboost_estimator : CatBoostClassifier or None
        Either a CatBoostClassifier with best found parameters or a fresh estimator you want to 
        store in the dictionary.

    Returns
    -------
    models : dict
        A dictionary mapping strings ("LogReg", "SVM", etc.) to either:
          - sklearn Pipeline or model
          - RandomizedSearchCV or GridSearchCV objects
          - CatBoostClassifier object
    """

    models = {}

    # 1) Logistic Regression with RandomizedSearchCV
    logreg_search = RandomizedSearchCV(
        Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=0.5,
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            ))
        ]),
        param_distributions={
            "clf__C":       [1e-3, 1e-2, 1e-1, 1, 10, 100],
            "clf__l1_ratio":[0.0, 0.25, 0.5, 0.75, 1.0]
        },
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )
    models["LogReg"] = logreg_search

    # 2) SVM with RandomizedSearchCV
    svm_search = RandomizedSearchCV(
        Pipeline([
            ("prep", preprocessor),
            ("clf", SVC(probability=True, kernel="rbf", class_weight="balanced"))
        ]),
        param_distributions={
            "clf__C":     np.logspace(-3, 3, 20),
            "clf__gamma": np.logspace(-4, 0, 20)
        },
        n_iter=30,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )
    models["SVM"] = svm_search

    # 3) Naive Bayes with GridSearchCV
    nb_grid = GridSearchCV(
        GaussianNB(),
        param_grid={"var_smoothing": np.logspace(-12, -6, 6)},
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )
    models["NaiveBayes"] = nb_grid

    # 4) ExtraTrees with known best params (if provided)
    if best_et_params is not None:
        et = ExtraTreesClassifier(
            **best_et_params,  # e.g. n_estimators=300, max_depth=10, etc.
            random_state=42,
            class_weight="balanced"
        )
        models["ExtraTrees"] = et

    # 5) HistGradientBoostingClassifier with known best params
    if best_hgb_params is not None:
        hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=best_hgb_params["learning_rate"],
            max_iter=best_hgb_params["max_iter"],
            max_depth=best_hgb_params["max_depth"],
            min_samples_leaf=best_hgb_params["min_samples_leaf"],
            l2_regularization=best_hgb_params["l2_regularization"],
            max_leaf_nodes=best_hgb_params["max_leaf_nodes"],
            early_stopping="auto",
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        models["HistGB"] = hgb

    # 6) AdaBoost with known best params
    if best_ab_params is not None:
        ab = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            algorithm="SAMME",
            n_estimators=best_ab_params["n_estimators"],
            learning_rate=best_ab_params["learning_rate"],
            random_state=42
        )
        models["AdaBoost"] = ab

    # 7) XGBoost with known best params
    if best_xgb_params is not None:
        xgb_clf = xgb.XGBClassifier(
            **best_xgb_params,         # e.g. n_estimators=500, max_depth=5, ...
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="auc",
            tree_method="hist",
            scale_pos_weight=1.0,      # adjust if needed for class imbalance
            random_state=42
        )
        models["XGBoost"] = xgb_clf

    # 8) CatBoost (e.g. from an external param search or a final model)
    if catboost_estimator is not None:
        models["CatBoost"] = catboost_estimator

    return models

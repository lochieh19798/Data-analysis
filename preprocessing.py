# preprocessing.py
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(numeric_features, categorical_features):
    """
    Returns a ColumnTransformer that:
      - KNN-imputes + scales numeric features
      - KNN-imputes + one-hot-encodes categorical features
    """
    numeric_pipe = Pipeline(steps=[
        ("impute", KNNImputer(n_neighbors=5, weights="uniform")),
        ("scale", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
        ("impute", KNNImputer(n_neighbors=5)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor

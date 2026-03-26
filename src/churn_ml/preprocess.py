import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["TotalCharges"] = pd.to_numeric(prepared["TotalCharges"], errors="coerce")
    prepared["Churn"] = prepared["Churn"].map({"Yes": 1, "No": 0})
    return prepared


def build_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]
    return X, y


def prepare_inference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "TotalCharges" in prepared.columns:
        prepared["TotalCharges"] = pd.to_numeric(prepared["TotalCharges"], errors="coerce")
    if "Churn" in prepared.columns:
        prepared = prepared.drop(columns=["Churn"])
    return prepared


def build_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [column for column in ["customerID"] if column in df.columns]
    return df.drop(columns=drop_columns)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

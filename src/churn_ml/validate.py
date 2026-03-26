import pandas as pd

from churn_ml.constants import ALLOWED_CATEGORIES, NUMERIC_COLUMNS, REQUIRED_COLUMNS


def _numeric_coercion_report(df: pd.DataFrame) -> dict:
    report = {}
    for column in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[column], errors="coerce")
        empty_strings = df[column].astype(str).str.strip().eq("")
        invalid_mask = coerced.isna() & df[column].notna() & ~empty_strings
        report[column] = {
            "invalid_non_numeric_count": int(invalid_mask.sum()),
            "coerced_missing_count": int(coerced.isna().sum()),
        }
    return report


def _categorical_sanity_report(df: pd.DataFrame) -> dict:
    report = {}
    for column, allowed_values in ALLOWED_CATEGORIES.items():
        observed_values = set(df[column].dropna().astype(str).unique())
        unexpected_values = sorted(observed_values - allowed_values)
        report[column] = {
            "unexpected_values": unexpected_values,
            "unexpected_count": int(df[column].astype(str).isin(unexpected_values).sum()) if unexpected_values else 0,
        }
    return report


def validate_dataset(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values found.")

    if df["Churn"].isna().any():
        raise ValueError("Target column contains null values.")

    numeric_report = _numeric_coercion_report(df)
    invalid_numeric_columns = [
        column
        for column, details in numeric_report.items()
        if details["invalid_non_numeric_count"] > 0
    ]
    if invalid_numeric_columns:
        raise ValueError(f"Invalid numeric values found in columns: {invalid_numeric_columns}")

    categorical_report = _categorical_sanity_report(df)
    unexpected_category_columns = [
        column
        for column, details in categorical_report.items()
        if details["unexpected_count"] > 0
    ]
    if unexpected_category_columns:
        raise ValueError(f"Unexpected categorical values found in columns: {unexpected_category_columns}")


def build_data_quality_report(df: pd.DataFrame) -> dict:
    total_rows = int(len(df))
    missing_counts = {column: int(df[column].isna().sum()) for column in df.columns}
    missing_rates = {
        column: round(float(df[column].isna().mean()), 4) for column in df.columns
    }
    numeric_report = _numeric_coercion_report(df)
    categorical_report = _categorical_sanity_report(df)

    return {
        "row_count": total_rows,
        "column_count": int(len(df.columns)),
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else None,
        "missing_values": missing_counts,
        "missing_rates": missing_rates,
        "numeric_coercion": numeric_report,
        "categorical_sanity": categorical_report,
        "target_distribution": (
            df["Churn"].value_counts(dropna=False).to_dict() if "Churn" in df.columns else {}
        ),
    }

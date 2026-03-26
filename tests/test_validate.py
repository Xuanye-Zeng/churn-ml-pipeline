import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from churn_ml.validate import build_data_quality_report, validate_dataset


def test_validate_dataset_rejects_duplicate_customer_ids():
    df = pd.DataFrame(
        {
            "customerID": ["1", "1"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 0],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 2],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Electronic check", "Electronic check"],
            "MonthlyCharges": [10.0, 20.0],
            "TotalCharges": ["10.0", "20.0"],
            "Churn": ["Yes", "No"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate customerID values found."):
        validate_dataset(df)


def test_data_quality_report_contains_basic_counts():
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "tiny_churn.csv"
    df = pd.read_csv(fixture_path)

    report = build_data_quality_report(df)

    assert report["row_count"] == 12
    assert report["column_count"] == 21
    assert report["duplicate_customer_ids"] == 0
    assert "missing_rates" in report
    assert "numeric_coercion" in report
    assert report["numeric_coercion"]["TotalCharges"]["invalid_non_numeric_count"] == 0


def test_validate_dataset_rejects_unexpected_target_values():
    df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 0],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 2],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Electronic check", "Electronic check"],
            "MonthlyCharges": [10.0, 20.0],
            "TotalCharges": ["10.0", "20.0"],
            "Churn": ["Maybe", "No"],
        }
    )

    with pytest.raises(ValueError, match="Unexpected categorical values found"):
        validate_dataset(df)

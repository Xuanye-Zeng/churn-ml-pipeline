import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from churn_ml.cli import run_pipeline
from churn_ml.predict import predict_batch


def test_run_pipeline_with_fixture_dataset(tmp_path):
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "tiny_churn.csv"
    config_path = tmp_path / "test_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_path": str(fixture_path),
                "output_dir": str(tmp_path / "outputs"),
                "torch": {
                    "enabled": True,
                    "epochs": 6,
                    "batch_size": 4,
                    "hidden_dims": [16, 8],
                    "validation_split": 0.25,
                    "early_stopping_patience": 2,
                },
            }
        )
    )

    result = run_pipeline(config_path=str(config_path))

    assert "logistic_regression" in result["metrics"]
    assert "random_forest" in result["metrics"]
    assert "torch_mlp" in result["metrics"]
    assert "cross_validation" in result["metrics"]["logistic_regression"]
    assert "cross_validation" in result["metrics"]["random_forest"]
    assert "training_summary" in result["metrics"]["torch_mlp"]
    assert result["metrics"]["torch_mlp"]["training_summary"]["epochs_completed"] >= 1
    assert result["threshold_report"]["best_f1_threshold"] >= 0.1
    assert Path(result["run_summary"]["best_model_artifact"]).exists()
    assert Path(result["run_summary"]["threshold_report_path"]).exists()
    assert Path(result["run_summary"]["manifest_path"]).exists()
    assert result["manifest"]["selected_model"] == result["run_summary"]["selected_model"]

    predictions_path = tmp_path / "predictions.csv"
    predict_batch(
        model_path=result["run_summary"]["best_model_artifact"],
        input_csv=str(fixture_path),
        output_csv=str(predictions_path),
    )

    predictions_df = pd.read_csv(predictions_path)
    assert "churn_probability" in predictions_df.columns
    assert "predicted_churn" in predictions_df.columns

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from churn_ml.artifacts import build_run_manifest, build_run_paths, save_model_artifact, write_json
from churn_ml.config import load_config
from churn_ml.evaluate import build_threshold_report
from churn_ml.ingest import download_dataset, ensure_directory, load_dataset
from churn_ml.preprocess import build_features_and_target, build_preprocessor, prepare_dataframe
from churn_ml.train_sklearn import train_sklearn_models
from churn_ml.train_torch import train_torch_model
from churn_ml.validate import build_data_quality_report, validate_dataset


def select_best_result(results: dict, metric_name: str) -> dict:
    best_name = max(results, key=lambda name: results[name]["metrics"][metric_name])
    return {"name": best_name, **results[best_name]}


def build_run_summary(prepared_df, X_train, X_test, y, best_result, model_results, run_paths, config, model_path):
    return {
        "dataset_rows": int(len(prepared_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_positive_rate": round(float(y.mean()), 4),
        "selected_model": best_result["name"],
        "selected_framework": best_result["framework"],
        "selection_metric": config["selection_metric"],
        "run_dir": str(run_paths.run_dir),
        "best_model_artifact": str(model_path),
        "threshold_report_path": str(run_paths.threshold_report_path),
        "manifest_path": str(run_paths.manifest_path),
        "models_evaluated": list(model_results.keys()),
    }


def run_pipeline(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    run_paths = build_run_paths(config["output_dir"])
    ensure_directory(run_paths.output_dir)

    if config.get("dataset_path"):
        source_dataset_path = Path(config["dataset_path"])
    else:
        source_dataset_path = download_dataset(config["data_url"], run_paths.dataset_path)

    # Copy dataset into the run directory (for traceability) and to the
    # output root (as a convenience pointer for batch prediction).
    if source_dataset_path.resolve() != run_paths.dataset_path.resolve():
        shutil.copy2(source_dataset_path, run_paths.dataset_path)
    shutil.copy2(source_dataset_path, run_paths.output_dir / "dataset.csv")

    raw_df = load_dataset(run_paths.dataset_path)
    validate_dataset(raw_df)
    data_quality_report = build_data_quality_report(raw_df)
    write_json(run_paths.data_quality_path, data_quality_report)

    prepared_df = prepare_dataframe(raw_df)
    X, y = build_features_and_target(prepared_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    model_results = train_sklearn_models(config, preprocessor, X_train, X_test, y_train, y_test)
    model_results.update(train_torch_model(config, preprocessor, X_train, X_test, y_train, y_test))

    metrics_payload = {}
    for name, details in model_results.items():
        payload = {"test_metrics": details["metrics"]}
        if details.get("cross_validation") is not None:
            payload["cross_validation"] = details["cross_validation"]
        if details.get("training_summary") is not None:
            payload["training_summary"] = details["training_summary"]
        metrics_payload[name] = payload

    best_result = select_best_result(model_results, config["selection_metric"])
    threshold_report = build_threshold_report(y_test, best_result["test_probabilities"])
    model_path = save_model_artifact(best_result, run_paths)
    run_summary = build_run_summary(
        prepared_df,
        X_train,
        X_test,
        y,
        best_result,
        model_results,
        run_paths,
        config,
        model_path,
    )
    run_manifest = build_run_manifest(config, run_paths, run_summary)

    write_json(run_paths.metrics_path, metrics_payload)
    write_json(run_paths.summary_path, run_summary)
    write_json(run_paths.threshold_report_path, threshold_report)
    write_json(run_paths.manifest_path, run_manifest)
    write_json(run_paths.latest_metrics_path, metrics_payload)
    write_json(run_paths.latest_summary_path, run_summary)
    write_json(run_paths.latest_threshold_report_path, threshold_report)
    write_json(run_paths.latest_manifest_path, run_manifest)

    return {
        "metrics": metrics_payload,
        "run_summary": run_summary,
        "threshold_report": threshold_report,
        "manifest": run_manifest,
        "run_dir": str(run_paths.run_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the customer churn ML pipeline.")
    parser.add_argument("--config", help="Path to a JSON config file.", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(config_path=args.config)
    print("Pipeline run completed.")
    print(f"Run directory: {result['run_dir']}")
    print(f"Selected model: {result['run_summary']['selected_model']}")
    print(f"Selected framework: {result['run_summary']['selected_framework']}")


if __name__ == "__main__":
    main()

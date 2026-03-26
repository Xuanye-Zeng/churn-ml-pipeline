import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shutil

import joblib
import torch


@dataclass
class RunPaths:
    output_dir: Path
    run_dir: Path
    dataset_path: Path
    metrics_path: Path
    summary_path: Path
    data_quality_path: Path
    threshold_report_path: Path
    manifest_path: Path
    selected_model_path: Path
    latest_metrics_path: Path
    latest_summary_path: Path
    latest_threshold_report_path: Path
    latest_manifest_path: Path


def build_run_paths(output_dir: str) -> RunPaths:
    base_output_dir = Path(output_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        output_dir=base_output_dir,
        run_dir=run_dir,
        dataset_path=run_dir / "dataset.csv",
        metrics_path=run_dir / "metrics.json",
        summary_path=run_dir / "run_summary.json",
        data_quality_path=run_dir / "data_quality_report.json",
        threshold_report_path=run_dir / "threshold_report.json",
        manifest_path=run_dir / "manifest.json",
        selected_model_path=run_dir / "best_model",
        latest_metrics_path=base_output_dir / "metrics.json",
        latest_summary_path=base_output_dir / "run_summary.json",
        latest_threshold_report_path=base_output_dir / "threshold_report.json",
        latest_manifest_path=base_output_dir / "manifest.json",
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def collect_dependency_versions() -> dict:
    packages = ["pandas", "numpy", "scikit-learn", "torch", "joblib", "requests"]
    versions = {}
    for package_name in packages:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions


def build_run_manifest(config: dict, run_paths: RunPaths, run_summary: dict) -> dict:
    return {
        "run_id": run_paths.run_dir.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "selected_model": run_summary["selected_model"],
        "selected_framework": run_summary["selected_framework"],
        "selection_metric": run_summary["selection_metric"],
        "artifact_paths": {
            "run_dir": str(run_paths.run_dir),
            "dataset": str(run_paths.dataset_path),
            "metrics": str(run_paths.metrics_path),
            "run_summary": str(run_paths.summary_path),
            "data_quality_report": str(run_paths.data_quality_path),
            "threshold_report": str(run_paths.threshold_report_path),
            "model": run_summary["best_model_artifact"],
        },
        "dependency_versions": collect_dependency_versions(),
    }


def save_model_artifact(result: dict, run_paths: RunPaths):
    artifact_type = result["artifact_type"]
    latest_joblib_path = run_paths.output_dir / "best_model.pkl"
    latest_torch_path = run_paths.output_dir / "best_model.pt"

    if artifact_type == "joblib":
        model_path = run_paths.selected_model_path.with_suffix(".pkl")
        joblib.dump(result["model_object"], model_path)
        if latest_torch_path.exists():
            latest_torch_path.unlink()
        shutil.copy2(model_path, latest_joblib_path)
        return model_path

    model_path = run_paths.selected_model_path.with_suffix(".pt")
    torch.save(
        {
            "model_state_dict": result["model_object"].state_dict(),
            "preprocessor": result["preprocessor"],
            "input_dim": result["input_dim"],
            "hidden_dims": result["hidden_dims"],
            "dropout": result["dropout"],
            "training_summary": result.get("training_summary"),
        },
        model_path,
    )
    if latest_joblib_path.exists():
        latest_joblib_path.unlink()
    shutil.copy2(model_path, latest_torch_path)
    return model_path

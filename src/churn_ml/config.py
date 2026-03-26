import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "data_url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "dataset_path": None,
    "output_dir": "outputs",
    "selection_metric": "f1",
    "test_size": 0.2,
    "random_state": 42,
    "torch": {
        "enabled": True,
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.001,
        "hidden_dims": [128, 64],
        "dropout": 0.1,
        "weight_decay": 0.0001,
    },
    "models": {
        "logistic_regression": {
            "enabled": True,
            "max_iter": 2000,
        },
        "random_forest": {
            "enabled": True,
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 2,
        },
    },
}


def deep_update(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)

    path = Path(config_path)
    overrides = json.loads(path.read_text())
    return deep_update(DEFAULT_CONFIG, overrides)

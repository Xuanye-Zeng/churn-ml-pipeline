from pathlib import Path

import pandas as pd
import requests


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_dataset(data_url: str, dataset_path: Path) -> Path:
    response = requests.get(data_url, timeout=30)
    response.raise_for_status()
    dataset_path.write_bytes(response.content)
    return dataset_path


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

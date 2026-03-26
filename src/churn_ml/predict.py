import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch

from churn_ml.evaluate import probabilities_to_predictions
from churn_ml.preprocess import build_inference_features, prepare_inference_dataframe
from churn_ml.train_torch import ChurnMLP, to_dense_array


def load_sklearn_bundle(model_path: Path):
    return {"framework": "scikit-learn", "model": joblib.load(model_path)}


def load_torch_bundle(model_path: Path):
    # weights_only=False because the checkpoint includes a sklearn preprocessor
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ChurnMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "framework": "pytorch",
        "model": model,
        "preprocessor": checkpoint["preprocessor"],
    }


def load_model_bundle(model_path: str):
    path = Path(model_path)
    if path.suffix == ".pkl":
        return load_sklearn_bundle(path)
    if path.suffix == ".pt":
        return load_torch_bundle(path)
    raise ValueError(f"Unsupported model artifact: {path}")


def predict_batch(model_path: str, input_csv: str, output_csv: str) -> Path:
    bundle = load_model_bundle(model_path)
    raw_df = pd.read_csv(input_csv)
    prepared_df = prepare_inference_dataframe(raw_df)
    features = build_inference_features(prepared_df)

    if bundle["framework"] == "scikit-learn":
        probabilities = bundle["model"].predict_proba(features)[:, 1]
        predictions = bundle["model"].predict(features)
    else:
        transformed = bundle["preprocessor"].transform(features)
        dense_features = to_dense_array(transformed)
        with torch.no_grad():
            logits = bundle["model"](torch.tensor(dense_features, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).numpy()
        predictions = probabilities_to_predictions(probabilities)

    output_df = raw_df.copy()
    output_df["churn_probability"] = probabilities
    output_df["predicted_churn"] = predictions

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch predictions with a saved churn model.")
    parser.add_argument("--model-path", required=True, help="Path to a saved .pkl or .pt model artifact.")
    parser.add_argument("--input-csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-csv", required=True, help="Path to save the prediction CSV output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = predict_batch(
        model_path=args.model_path,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
    )
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()

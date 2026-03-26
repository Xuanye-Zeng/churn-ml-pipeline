import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import clone

from churn_ml.evaluate import compute_metrics, probabilities_to_predictions


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs).squeeze(1)


def to_dense_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _split_torch_training_data(config: dict, X_train, y_train):
    validation_split = config["torch"].get("validation_split", 0.2)
    class_counts = y_train.value_counts()
    can_stratify = len(class_counts) > 1 and int(class_counts.min()) >= 2

    if validation_split <= 0 or not can_stratify:
        return X_train, None, y_train, None

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_split,
        random_state=config["random_state"],
        stratify=y_train,
    )
    return X_train_sub, X_val, y_train_sub, y_val


def train_torch_model(config: dict, preprocessor, X_train, X_test, y_train, y_test):
    torch_config = config["torch"]
    if not torch_config.get("enabled", True):
        return {}

    set_torch_seed(config["random_state"])
    X_train_sub, X_val, y_train_sub, y_val = _split_torch_training_data(config, X_train, y_train)
    fitted_preprocessor = clone(preprocessor)
    X_train_transformed = to_dense_array(fitted_preprocessor.fit_transform(X_train_sub))
    X_test_transformed = to_dense_array(fitted_preprocessor.transform(X_test))
    X_val_transformed = (
        to_dense_array(fitted_preprocessor.transform(X_val)) if X_val is not None else None
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_transformed, dtype=torch.float32),
        torch.tensor(y_train_sub.to_numpy(), dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=torch_config.get("batch_size", 128),
        shuffle=True,
    )

    model = ChurnMLP(
        input_dim=X_train_transformed.shape[1],
        hidden_dims=torch_config.get("hidden_dims", [128, 64]),
        dropout=torch_config.get("dropout", 0.1),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=torch_config.get("learning_rate", 0.001),
        weight_decay=torch_config.get("weight_decay", 0.0001),
    )
    criterion = nn.BCEWithLogitsLoss()
    patience = torch_config.get("early_stopping_patience", 4)
    min_delta = torch_config.get("min_delta", 0.0005)
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    epochs_without_improvement = 0
    train_history = []

    for epoch in range(1, torch_config.get("epochs", 20) + 1):
        model.train()
        total_train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item()) * len(batch_labels)

        average_train_loss = total_train_loss / max(len(train_dataset), 1)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(average_train_loss), 6),
        }

        if X_val_transformed is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val_transformed, dtype=torch.float32))
                val_labels = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
                val_loss = criterion(val_logits, val_labels).item()
            epoch_record["val_loss"] = round(float(val_loss), 6)

            if val_loss + min_delta < best_val_loss:
                best_val_loss = float(val_loss)
                best_epoch = epoch
                best_state_dict = {
                    name: tensor.clone() for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    train_history.append(epoch_record)
                    break
        else:
            best_epoch = epoch
            best_state_dict = {
                name: tensor.clone() for name, tensor in model.state_dict().items()
            }

        train_history.append(epoch_record)

    model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_transformed, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).numpy()

    predictions = probabilities_to_predictions(probabilities)
    metrics = compute_metrics(y_test, predictions, probabilities)

    return {
        "torch_mlp": {
            "framework": "pytorch",
            "metrics": metrics,
            "artifact_type": "torch",
            "model_object": model,
            "preprocessor": fitted_preprocessor,
            "input_dim": int(X_train_transformed.shape[1]),
            "hidden_dims": torch_config.get("hidden_dims", [128, 64]),
            "dropout": torch_config.get("dropout", 0.1),
            "test_probabilities": probabilities.tolist(),
            "training_summary": {
                "train_rows": int(len(X_train_sub)),
                "validation_rows": int(len(X_val)) if X_val is not None else 0,
                "best_epoch": int(best_epoch),
                "epochs_completed": int(len(train_history)),
                "used_validation_split": X_val is not None,
                "history": train_history,
            },
        }
    }

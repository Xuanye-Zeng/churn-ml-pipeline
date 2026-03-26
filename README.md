# 🚀 Customer Churn ML Pipeline

A compact end-to-end training pipeline for customer churn prediction using the IBM Telco Customer Churn dataset.

The project is intentionally small, but structured like a real training workflow: data ingestion, validation, preprocessing, model training, evaluation, artifact generation, and batch prediction.

## 🚀 At A Glance

```mermaid
flowchart LR
configStep[ConfigAndCLI] --> ingestStep[LoadOrDownloadDataset]
ingestStep --> validateStep[ValidateAndProfileData]
validateStep --> prepStep[PreprocessFeatures]
prepStep --> splitStep[TrainTestSplit]

splitStep --> skStep[TrainSklearnModels]
splitStep --> torchStep[TrainTorchMLP]

skStep --> compareStep[CompareMetricsAndSelectModel]
torchStep --> compareStep

compareStep --> metricsStep[WriteMetricsAndReports]
metricsStep --> modelStep[SaveSelectedModel]
modelStep --> predictStep[BatchPredictionEntry]
```

- dataset: IBM Telco Customer Churn
- task: binary classification
- target: `Churn`
- models: `logistic_regression`, `random_forest`, `torch_mlp`
- selection metric: `f1`

## 🧰 What The Repository Covers

- config-driven training runs
- schema and data-quality validation
- shared preprocessing for numeric and categorical features
- sklearn and PyTorch baseline comparison
- cross-validation and threshold analysis
- per-run artifacts and run manifest
- simple batch prediction entrypoint
- tests, CI, and Docker

## 📊 Current Result

The latest run selects `logistic_regression` as the best model by held-out test `f1`.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `logistic_regression` | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| `random_forest` | 0.8055 | 0.6799 | 0.5053 | 0.5798 | 0.8423 |
| `torch_mlp` | 0.8020 | 0.6471 | 0.5588 | 0.5997 | 0.8395 |

Evaluation notes:

- `logistic_regression` also leads the sklearn baselines on cross-validation F1 mean
- the best F1 threshold for the selected model is `0.25`, compared with the default `0.5`
- lowering the threshold improves recall and slightly improves F1, while reducing precision and accuracy

## 📁 Repository Structure

```text
customer-churn-ml-pipeline/
  README.md
  technical_design.md
  requirements.txt
  .gitignore
  Dockerfile
  configs/
    default.json
  src/
    pipeline.py
    churn_ml/
      cli.py
      config.py
      ingest.py
      predict.py
      preprocess.py
      train_sklearn.py
      train_torch.py
      validate.py
  outputs/
    .gitkeep
    manifest.json
    metrics.json
    run_summary.json
    threshold_report.json
    runs/
  tests/
    fixtures/
      tiny_churn.csv
  .github/
    workflows/
      ci.yml
```

## ▶️ Run Training

```bash
.venv/bin/python src/pipeline.py --config configs/default.json
```

Equivalent:

```bash
make train
```

For a quick local smoke run without downloading the full dataset:

```bash
.venv/bin/python src/pipeline.py --config configs/smoke_test.json
```

Or:

```bash
make smoke
```

Top-level outputs:

- `outputs/metrics.json`
- `outputs/run_summary.json`
- `outputs/threshold_report.json`
- `outputs/manifest.json`
- `outputs/best_model.pkl` or `outputs/best_model.pt`

Each run also writes a timestamped directory under `outputs/runs/<run_id>/`.

## 🔮 Run Batch Prediction

```bash
.venv/bin/python src/predict.py --model-path outputs/best_model.pkl --input-csv outputs/dataset.csv --output-csv outputs/predictions.csv
```

Equivalent:

```bash
make predict
```

Prediction output adds:

- `churn_probability`
- `predicted_churn`

## 🔧 Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Equivalent:

```bash
make setup
```

## 🧪 Run Tests

```bash
.venv/bin/python -m pytest tests
```

Equivalent:

```bash
make test
```

## 📦 Docker

```bash
docker build -t churn-pipeline .
docker run --rm churn-pipeline
```

## ⚙️ CI

The repository includes a GitHub Actions workflow in `.github/workflows/ci.yml` that installs dependencies and runs the test suite on pushes and pull requests.

## 📓 Technical Design

The full design write-up is in [`technical_design.md`](./technical_design.md).

## ⚖️ Main Trade-off

The main trade-off is `speed of iteration vs reproducibility and modularity`.

A notebook or a single script would have been faster to build. This repository chooses a more structured path instead: explicit stages, repeatable artifacts, stronger evaluation outputs, and a prediction path that reuses saved model artifacts.

## 🧭 Possible Extensions

If expanded further, the next practical improvements would be:

- lightweight experiment tracking
- local Ray-based hyperparameter tuning
- richer data quality and drift monitoring
- fuller model serving or API-based inference

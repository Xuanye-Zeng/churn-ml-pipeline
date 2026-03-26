PYTHON := .venv/bin/python

.PHONY: setup train smoke test predict

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) src/pipeline.py --config configs/default.json

smoke:
	$(PYTHON) src/pipeline.py --config configs/smoke_test.json

test:
	$(PYTHON) -m pytest tests

predict:
	$(PYTHON) src/predict.py --model-path outputs/best_model.pkl --input-csv outputs/dataset.csv --output-csv outputs/predictions.csv

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src

CMD ["python", "src/pipeline.py", "--config", "configs/default.json"]

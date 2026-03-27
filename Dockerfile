# Stage 1: install dependencies in a builder layer
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: copy only installed packages + application code
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY configs ./configs
COPY src ./src

CMD ["python", "src/pipeline.py", "--config", "configs/default.json"]

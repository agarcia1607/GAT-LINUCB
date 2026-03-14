FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p \
    /app/artifacts/tickers \
    /app/artifacts/snapshots \
    /app/artifacts/X_raw \
    /app/artifacts/embeddings_gat \
    /app/artifacts/linucb \
    /app/data/raw \
    /app/data/processed \
    /app/logs \
    /app/reports \
    /app/outputs/manifests

CMD ["python", "run_pipeline.py"]
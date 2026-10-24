# Build from the repository root (the parent package `pricing` is needed):
#   docker build -f vol-surface-risk/Dockerfile -t volrisk .
#   docker run -p 8000:8000 volrisk
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Parent library first (changes rarely), then the service.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY vol-surface-risk/pyproject.toml ./vol-surface-risk/pyproject.toml
COPY vol-surface-risk/src ./vol-surface-risk/src
COPY vol-surface-risk/data/sample_SPY.csv vol-surface-risk/data/sample_SPY.json ./vol-surface-risk/data/
RUN pip install ./vol-surface-risk

# Serve the synthetic sample unless a data volume with real snapshots is mounted
# at /app/vol-surface-risk/data (-v $PWD/vol-surface-risk/data:/app/vol-surface-risk/data -e VOLRISK_SOURCE=snapshot).
ENV VOLRISK_DATA_DIR=/app/vol-surface-risk/data VOLRISK_SOURCE=sample
EXPOSE 8000
CMD ["uvicorn", "volrisk.api:app", "--host", "0.0.0.0", "--port", "8000"]

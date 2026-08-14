# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PIP_NO_CACHE_DIR=1 \
    VISIONGUARD_DEVICE=cpu \
    VISIONGUARD_MODEL_BACKEND=dummy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY api ./api
COPY ui ./ui
COPY configs ./configs
COPY models ./models
COPY assets ./assets

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install \
        opencv-python-headless \
        numpy \
        pillow \
        pyyaml \
        pydantic \
        pydantic-settings \
        fastapi \
        "uvicorn[standard]" \
        python-multipart \
        typer \
        rich \
        httpx \
        jinja2 \
    && pip install --no-deps -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

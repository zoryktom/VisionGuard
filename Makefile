.PHONY: help install install-dev lint format test test-cov run serve demo train export docker-build docker-up clean notebook

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
export PYTHONPATH := src:.

help:
	@echo "VisionGuard — real-time visual hazard detection"
	@echo ""
	@echo "  make install       Install runtime dependencies"
	@echo "  make install-dev   Install runtime + dev/test extras"
	@echo "  make lint          Ruff + mypy"
	@echo "  make format        Ruff format"
	@echo "  make test          Run unit tests"
	@echo "  make demo          Generate synthetic data and run dummy inference"
	@echo "  make serve         Launch FastAPI + dashboard"
	@echo "  make train         Train on the synthetic dataset"
	@echo "  make export        Export the latest checkpoint to ONNX"
	@echo "  make docker-up     Build and run the API container"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,train,export,ui]"
	$(PIP) install -r requirements-dev.txt

lint:
	ruff check src api tests scripts
	mypy src/visionguard

format:
	ruff format src api tests scripts
	ruff check --fix src api tests scripts

test:
	pytest tests -q

test-cov:
	pytest tests --cov --cov-report=term-missing --cov-report=xml

run: serve

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

demo:
	$(PYTHON) scripts/generate_synthetic_data.py --frames 40 --out data/datasets/synthetic
	$(PYTHON) -m visionguard detect --source data/datasets/synthetic/images --backend dummy --headless

train:
	$(PYTHON) -m visionguard train --config configs/training.yaml

export:
	$(PYTHON) -m visionguard export --config configs/training.yaml --format onnx

notebook:
	jupyter lab notebooks/01_explore_visionguard.ipynb

docker-build:
	docker build -t visionguard:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: help install install-dev setup test lint format clean docker-build docker-up docker-down

# Default target
help:
	@echo "NeuroSync Development Commands"
	@echo "============================="
	@echo "install       - Install production dependencies"
	@echo "install-dev   - Install development dependencies"
	@echo "setup         - Complete development setup"
	@echo "test          - Run tests"
	@echo "lint          - Run linting"
	@echo "format        - Format code"
	@echo "clean         - Clean temporary files"
	@echo "docker-build  - Build Docker images"
	@echo "docker-up     - Start Docker services"
	@echo "docker-down   - Stop Docker services"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

# Download the spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data (needed for some tokenizers)
python -m nltk.downloader punkt

# Complete development setup
setup: install-dev
	pre-commit install
	@echo "Creating data directories..."
	mkdir -p data logs uploads data/vector_store
	@echo "Setup complete! Run 'make docker-up' to start services."

# Run tests
test:
	pytest tests/ -v --cov=src/neurosync

# Run linting
lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf *.egg-info

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Development workflow
dev-setup: setup
	@echo "Starting development environment..."
	docker-compose up -d postgres redis
	@echo "Development environment ready!"

# Full integration test
integration-test: docker-up
	sleep 10  # Wait for services to start
	pytest tests/integration/ -v
	docker-compose down

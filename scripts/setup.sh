#!/bin/bash

set -e

echo "Setting up Neurosync environment..."

# Check Python version
PYTHON_CMD="/opt/homebrew/bin/python3.11"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Python 3.11 not found at $PYTHON_CMD"
    echo "Please install Python 3.11: brew install python@3.11"
    exit 1
fi

python_version=$($PYTHON_CMD --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.11"

echo "Using Python version: $python_version"

echo "Python version check passed"

# Create virtual environment with Python 3.11
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
echo "Installing NeuroSync in development mode..."
pip install -e .

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs uploads data/vector_store

# Copy environment file
if [ ! -f ".env" ]; then
    echo "Creating environment file..."
    cp .env.example .env
    echo "Please update .env file with your configuration"
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "Docker is available"
    if command -v docker-compose &> /dev/null; then
        echo "Docker Compose is available"
        echo "Building Docker images..."
        docker-compose build --no-cache
    else
        echo "Docker Compose not found. Please install Docker Compose."
    fi
else
    echo "Docker not found. Please install Docker."
fi

echo ""
echo "NeuroSync development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Run 'make docker-up' to start services"
echo "3. Run 'neurosync --help' to see available commands"
echo "4. Visit http://localhost:8000 for the API"
echo "5. Visit http://localhost:8080 for Airflow UI"

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .
COPY docker/client-cli/requirements.txt ./docker-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r docker-requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Install the package in development mode
RUN pip install -e .

# Default command
CMD ["neurosync", "--help"]

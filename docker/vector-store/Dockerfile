# Vector Store Dockerfile (using FAISS)
FROM python:3-slim

# Install dependencies for FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install faiss-cpu numpy

# Create app directory
WORKDIR /app

# Create volume for vector data
VOLUME /app/data

EXPOSE 8080

CMD ["python", "-c", "print('Vector store service ready')"]

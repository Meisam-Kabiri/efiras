FROM python:3.10-slim

WORKDIR /app

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the model to avoid startup delays
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

COPY . .

# Create directories that your app expects
RUN mkdir -p indexes data_processed

# Use PORT environment variable from Cloud Run
CMD uvicorn fastapi_backend:app --host 0.0.0.0 --port $PORT
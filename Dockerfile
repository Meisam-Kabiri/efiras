FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including curl for downloads
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the model to avoid startup delays
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"


# Create indexes directory
RUN mkdir -p indexes data_processed
# Download index files from public S3 URLs
RUN echo "📥 Downloading BM25 index (smallest file first)..." && \
    curl -f -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/bm25_tokenized.pkl" -o indexes/bm25_tokenized.pkl && \
    echo "✅ Downloaded BM25 index (20MB)" && \
    \
    echo "📥 Downloading FAISS index..." && \
    curl -f -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/faiss.index" -o indexes/faiss.index && \
    echo "✅ Downloaded FAISS index (160MB)" && \
    \
    echo "📥 Downloading chunks metadata (largest file last)..." && \
    curl -f -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/chunks_metadata.json" -o indexes/chunks_metadata.json && \
    echo "✅ Downloaded chunks metadata (1.3GB)"

# Only copy code AFTER successful downloads
COPY . .

# Verify files were downloaded successfully
RUN ls -la indexes/ && \
    echo "Downloaded files:" && \
    du -h indexes/*

# Use PORT environment variable (works for Railway, Cloud Run, etc.)
CMD uvicorn fastapi_backend:app --host 0.0.0.0 --port $PORT
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

# Download index files in order from smallest to largest to catch errors early
# This way if there's a problem with URLs/permissions, we fail fast
RUN echo "📥 Downloading BM25 index (smallest file first)..." && \
    curl -f -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/bm25_tokenized.pkl?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIASVQKHXMD42MIU3RR%2F20250801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250801T132848Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCICfggoacVk7Y%2FUAVnqDXaQZ2TxQxDrYY051UW5797xlWAiEAnXfahswxxuTdzDF8zFxAMugHPGOSaZEVD%2B5WDGLV67MqgQMI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwxODM2MzEzMzAwNTUiDGDyFdKyeErIR%2B5fICrVAi5YX3Oafh3%2ButdsxgCM4Mc73EVgFN7%2FU3O%2FZemcoGhHPXFZJhtbgRQjvjWdFMHVnwn%2F7iZKCNj0dDOhAwRQlFn21xOmMcsBMVBUYxjJcz5lcRC0kIMdWqCT2SQikvkCata3BX6IaDsDoJXONzUFAdS%2B9Vm60UtSUCIHQ4I9ZuaXGyoOrsuSW10mkVcIdEfWRacks7EbWJOB9gpx9PNgMlhZDuk1Y9XSP3mmSPsntGS%2FCcIq1%2FprdPThFJbB9kwM9xw5jSHs%2Bua27rdMCXA5utSeGvf9QkPSZm928uBDH5jYNICz6WuwBv9sfk1gu%2FrpcM4bCCU7yYYE0KMmWYyKDPHwMewV1XXyzGu9Ycvkd9yUmtQPZyOtvSJoLrqAlQBoUdA9JvGVhJ%2F7HLAwOh22DEMlk5gszF40kfHwgYqRnxGVNoKF0%2BgwThyfVleIYhTK%2FpRYiEESMIjxssQGOq0C9LNs6H%2FgY27QyK5RSYhoHCPbz7SYcVlJ5VxGKVPfUEgEyK4PBVLPJW6zVtdl2qb9Xrrw0P57HcXEBBaaaYOabuMeeKzQbJV10VEZP5I%2BUWURmAg3Phvpy7c343o2LCPdkW5YnPDQmYKm0NDGLL0xLQy7Vl4NXEn5sRhpcukJNUZEW3h1CCy9dn2l4zSnJeqb5Zr6CYBrQabgRXAU6DjC8u7enF8wI%2BbWPsajV0uXK%2BxRFY%2Bus8%2Fi8sY6e3mPPLjm258rnLQVdfPpcmZArKbrhy%2FvPj1uY1Naw8aJjqggO3Z5y%2BPGf5jbjKGlYE2o6zk%2FUcA148fp8ZEN7M2uDM7%2BSJ4Qz6vvGCAY69%2B1f25eiCJepszjQIdTTNZtepdefmwlXBvyYY5CSbB0TaZnCQ%3D%3D&X-Amz-Signature=1213af1fed45f48fbb66e46a3ea42ab642f9820b4e2e5a532be1e7a9d814d0b7" -o indexes/bm25_tokenized.pkl && \
    echo "✅ Downloaded BM25 index (20MB)" && \
    \
    echo "📥 Downloading FAISS index..." && \
    curl -f -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/faiss.index?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIASVQKHXMD42MIU3RR%2F20250801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250801T132942Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCICfggoacVk7Y%2FUAVnqDXaQZ2TxQxDrYY051UW5797xlWAiEAnXfahswxxuTdzDF8zFxAMugHPGOSaZEVD%2B5WDGLV67MqgQMI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwxODM2MzEzMzAwNTUiDGDyFdKyeErIR%2B5fICrVAi5YX3Oafh3%2ButdsxgCM4Mc73EVgFN7%2FU3O%2FZemcoGhHPXFZJhtbgRQjvjWdFMHVnwn%2F7iZKCNj0dDOhAwRQlFn21xOmMcsBMVBUYxjJcz5lcRC0kIMdWqCT2SQikvkCata3BX6IaDsDoJXONzUFAdS%2B9Vm60UtSUCIHQ4I9ZuaXGyoOrsuSW10mkVcIdEfWRacks7EbWJOB9gpx9PNgMlhZDuk1Y9XSP3mmSPsntGS%2FCcIq1%2FprdPThFJbB9kwM9xw5jSHs%2Bua27rdMCXA5utSeGvf9QkPSZm928uBDH5jYNICz6WuwBv9sfk1gu%2FrpcM4bCCU7yYYE0KMmWYyKDPHwMewV1XXyzGu9Ycvkd9yUmtQPZyOtvSJoLrqAlQBoUdA9JvGVhJ%2F7HLAwOh22DEMlk5gszF40kfHwgYqRnxGVNoKF0%2BgwThyfVleIYhTK%2FpRYiEESMIjxssQGOq0C9LNs6H%2FgY27QyK5RSYhoHCPbz7SYcVlJ5VxGKVPfUEgEyK4PBVLPJW6zVtdl2qb9Xrrw0P57HcXEBBaaaYOabuMeeKzQbJV10VEZP5I%2BUWURmAg3Phvpy7c343o2LCPdkW5YnPDQmYKm0NDGLL0xLQy7Vl4NXEn5sRhpcukJNUZEW3h1CCy9dn2l4zSnJeqb5Zr6CYBrQabgRXAU6DjC8u7enF8wI%2BbWPsajV0uXK%2BxRFY%2Bus8%2Fi8sY6e3mPPLjm258rnLQVdfPpcmZArKbrhy%2FvPj1uY1Naw8aJjqggO3Z5y%2BPGf5jbjKGlYE2o6zk%2FUcA148fp8ZEN7M2uDM7%2BSJ4Qz6vvGCAY69%2B1f25eiCJepszjQIdTTNZtepdefmwlXBvyYY5CSbB0TaZnCQ%3D%3D&X-Amz-Signature=4d0e205fb97053167a92bcb73b152c655e4bee14f144d86f1058594462bd0228" -o indexes/faiss.index && \
    echo "✅ Downloaded FAISS index (160MB)" && \
    \
    echo "📥 Downloading chunks metadata (largest file last)..." && \
    curl -f -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/chunks_metadata.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIASVQKHXMD42MIU3RR%2F20250801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250801T133040Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCICfggoacVk7Y%2FUAVnqDXaQZ2TxQxDrYY051UW5797xlWAiEAnXfahswxxuTdzDF8zFxAMugHPGOSaZEVD%2B5WDGLV67MqgQMI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwxODM2MzEzMzAwNTUiDGDyFdKyeErIR%2B5fICrVAi5YX3Oafh3%2ButdsxgCM4Mc73EVgFN7%2FU3O%2FZemcoGhHPXFZJhtbgRQjvjWdFMHVnwn%2F7iZKCNj0dDOhAwRQlFn21xOmMcsBMVBUYxjJcz5lcRC0kIMdWqCT2SQikvkCata3BX6IaDsDoJXONzUFAdS%2B9Vm60UtSUCIHQ4I9ZuaXGyoOrsuSW10mkVcIdEfWRacks7EbWJOB9gpx9PNgMlhZDuk1Y9XSP3mmSPsntGS%2FCcIq1%2FprdPThFJbB9kwM9xw5jSHs%2Bua27rdMCXA5utSeGvf9QkPSZm928uBDH5jYNICz6WuwBv9sfk1gu%2FrpcM4bCCU7yYYE0KMmWYyKDPHwMewV1XXyzGu9Ycvkd9yUmtQPZyOtvSJoLrqAlQBoUdA9JvGVhJ%2F7HLAwOh22DEMlk5gszF40kfHwgYqRnxGVNoKF0%2BgwThyfVleIYhTK%2FpRYiEESMIjxssQGOq0C9LNs6H%2FgY27QyK5RSYhoHCPbz7SYcVlJ5VxGKVPfUEgEyK4PBVLPJW6zVtdl2qb9Xrrw0P57HcXEBBaaaYOabuMeeKzQbJV10VEZP5I%2BUWURmAg3Phvpy7c343o2LCPdkW5YnPDQmYKm0NDGLL0xLQy7Vl4NXEn5sRhpcukJNUZEW3h1CCy9dn2l4zSnJeqb5Zr6CYBrQabgRXAU6DjC8u7enF8wI%2BbWPsajV0uXK%2BxRFY%2Bus8%2Fi8sY6e3mPPLjm258rnLQVdfPpcmZArKbrhy%2FvPj1uY1Naw8aJjqggO3Z5y%2BPGf5jbjKGlYE2o6zk%2FUcA148fp8ZEN7M2uDM7%2BSJ4Qz6vvGCAY69%2B1f25eiCJepszjQIdTTNZtepdefmwlXBvyYY5CSbB0TaZnCQ%3D%3D&X-Amz-Signature=55e425144fbaafe14cb5070333329f3f3e9bc6e517d941f252b352c6cf42ffc4" -o indexes/chunks_metadata.json && \
    echo "✅ Downloaded chunks metadata (1.3GB)"

# Only copy code AFTER successful downloads
COPY . .

# Verify files were downloaded successfully
RUN ls -la indexes/ && \
    echo "Downloaded files:" && \
    du -h indexes/*

# Use PORT environment variable (works for Railway, Cloud Run, etc.)
CMD uvicorn fastapi_backend:app --host 0.0.0.0 --port $PORT
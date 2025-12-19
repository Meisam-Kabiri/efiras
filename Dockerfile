FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories (will be filled by warmup)
RUN mkdir -p data/indexes logs temps

# Cloud Run will set PORT environment variable
ENV PORT=8080
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Run uvicorn directly with the correct module path
# Use sh -c to allow PORT environment variable expansion
CMD ["sh", "-c", "uvicorn app.efiras_app:app --host 0.0.0.0 --port ${PORT}"]
# Stage 1: Builder stage to compile & install Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /srv

# Install build tools temporarily for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy & install lightweight production requirements directly into system site-packages
COPY requirements-prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Ultra-lightweight Production Runtime Image
FROM python:3.11-slim AS runner

WORKDIR /srv

# Copy installed site-packages and binaries from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment variables
ENV PYTHONPATH=/srv:/srv/app
ENV PORT=8080
EXPOSE 8080

# Copy application source code
COPY app/ app/
COPY core/ core/
COPY auth/ auth/

# Copy pre-built indexes
COPY data/regulatory_indexes/ data/regulatory_indexes/

CMD ["python", "-m", "uvicorn", "app.efiras_app:app", "--host", "0.0.0.0", "--port", "8080"]

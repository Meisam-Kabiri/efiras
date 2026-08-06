# Stage 1: Builder stage to compile & install Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /srv

# Install build tools temporarily for compilation if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy & install lightweight production requirements
COPY requirements-prod.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Ultra-lightweight Production Runtime Image (No build tools!)
FROM python:3.11-slim AS runner

WORKDIR /srv

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
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

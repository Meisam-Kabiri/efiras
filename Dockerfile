FROM python:3.11-slim

WORKDIR /srv

# System deps needed by wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use ultra-lightweight production requirements (~150MB, no PyTorch/Transformers)
COPY requirements-prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ app/
COPY core/ core/
COPY auth/ auth/

# Copy ONLY the pre-built indexes (regulatory_faiss.bin & regulatory_chunks.db)
COPY data/regulatory_indexes/ data/regulatory_indexes/

ENV PYTHONPATH=/srv:/srv/app
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.efiras_app:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.11-slim

WORKDIR /srv

# System deps needed by faiss/torch/pymupdf wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY core/ core/
COPY auth/ auth/
COPY data/indexes/faiss.index data/indexes/chunks.db data/indexes/

# "config", "endpoints", etc. are imported bare (e.g. `from config import ...`)
# inside app/efiras_app.py, so app/ itself must be on the path alongside the repo root.
ENV PYTHONPATH=/srv:/srv/app
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.efiras_app:app", "--host", "0.0.0.0", "--port", "8080"]

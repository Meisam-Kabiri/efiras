FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the model (saves it inside the container)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "fastapi_backend:app", "--host", "0.0.0.0", "--port", "8080"]
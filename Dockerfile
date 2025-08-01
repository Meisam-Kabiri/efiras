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

COPY . .

# Create indexes directory
RUN mkdir -p indexes data_processed

# Download index files in order from smallest to largest to catch errors early
# This way if there's a problem with URLs/permissions, we fail fast
RUN echo "📥 Downloading BM25 index (smallest file first)..." && \
    curl -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/bm25_tokenized.pkl?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIASVQKHXMDX32Z6GYL%2F20250801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250801T130450Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIA93iDQUP7gOU1mn%2B6ve2Tt1nfElvl0g3yzSQmAb5APyAiEAztEGSPvTpdodil%2FqTNfHwnWaj%2BV3gRLIvwHr%2FRkIRhsqgQMI7v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwxODM2MzEzMzAwNTUiDM9KnkoaW061F46cDCrVAng7NpVXswPWitWSucJWsEkSRY0nq96DZgwx0TGzMnzCYrZiZ0HUcs7t3ytsWB3%2BpjQMOKdzHDCnUkdngUt89QiIQxlo3eLleAXBl9sfafvIkmBXh4cDs3T1OJs44Dh%2BoOpkMfm1XM5bkBcoic78oHsHkST4mq2KSC9SMZc%2Ffza865sgro0c7Fw6TvqvsVaqmma7i368zk%2FQV80ciLtO2nPUGGdbSvcVy3MxncDzoafFl71yNQ8KMNUybECai8Lw%2Fvz8FJeQlu4gKPTA%2F8k%2FKGi8vg7EK%2Byp3OOAJ4rTPwTEdQfAdU5ULuYtUBYKnVytvKEIZJzi01l4L9f%2BMNksst3F2WvMdXtKhJQJgUPsgpRem0GP04zTuxid2ZCX0OUh7LpKZ3NwLMFe0yt1mvGZducIFwFzcHZMPQmTxusJtlY2P55rHTVvhr8izSIqpkE0ZOmfvPqqMIjxssQGOq0C5rs6RBzaq0yalBHnbW3rzzXUkNFSM%2Bezx%2BudbJ%2BzPtjTu49rHWrb%2FuaRB%2FBAx%2BEE1XpacHsPsrYSAqPuvTD7IoO2Rt4qBq4UcszJ7U3UguTdRmcWKvLZ43viw%2Fc%2BMmOwGQuU%2BAIeMDOfYw58r3ArSk9owHUUShR4LiF8ZBh0TYoCBwpzjA8PU1VF4yWUk%2FkKhcHE1AnhVnx9Ok97aXjgDlcy%2FBJBOGOrAWlDTVjfcUuMOY2ZXjMByX8xBZNvz0QlJhGrImFAwnDuVREe5UHPZn0UKC0ASbtq4WkNcQGNCPrmpmU3O9Ns47Bz64nTzKgdjmN7X77RgMKI0TVcWou2XO%2FcNmMhQaxaomrt0skbdomNXKLry508WLjoUufbyBGjBniPK%2B1XrhngvyWtSQ%3D%3D&X-Amz-Signature=14f5a4b91038750ad5b2a607bc81da5220fb965588772907aebfe2f3ddbdb622" -o indexes/bm25_tokenized.pkl && \
    echo "✅ Downloaded BM25 index (20MB)" && \
    \
    echo "📥 Downloading FAISS index..." && \
    curl -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/faiss.index?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIASVQKHXMDX32Z6GYL%2F20250801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250801T130452Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIA93iDQUP7gOU1mn%2B6ve2Tt1nfElvl0g3yzSQmAb5APyAiEAztEGSPvTpdodil%2FqTNfHwnWaj%2BV3gRLIvwHr%2FRkIRhsqgQMI7v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwxODM2MzEzMzAwNTUiDM9KnkoaW061F46cDCrVAng7NpVXswPWitWSucJWsEkSRY0nq96DZgwx0TGzMnzCYrZiZ0HUcs7t3ytsWB3%2BpjQMOKdzHDCnUkdngUt89QiIQxlo3eLleAXBl9sfafvIkmBXh4cDs3T1OJs44Dh%2BoOpkMfm1XM5bkBcoic78oHsHkST4mq2KSC9SMZc%2Ffza865sgro0c7Fw6TvqvsVaqmma7i368zk%2FQV80ciLtO2nPUGGdbSvcVy3MxncDzoafFl71yNQ8KMNUybECai8Lw%2Fvz8FJeQlu4gKPTA%2F8k%2FKGi8vg7EK%2Byp3OOAJ4rTPwTEdQfAdU5ULuYtUBYKnVytvKEIZJzi01l4L9f%2BMNksst3F2WvMdXtKhJQJgUPsgpRem0GP04zTuxid2ZCX0OUh7LpKZ3NwLMFe0yt1mvGZducIFwFzcHZMPQmTxusJtlY2P55rHTVvhr8izSIqpkE0ZOmfvPqqMIjxssQGOq0C5rs6RBzaq0yalBHnbW3rzzXUkNFSM%2Bezx%2BudbJ%2BzPtjTu49rHWrb%2FuaRB%2FBAx%2BEE1XpacHsPsrYSAqPuvTD7IoO2Rt4qBq4UcszJ7U3UguTdRmcWKvLZ43viw%2Fc%2BMmOwGQuU%2BAIeMDOfYw58r3ArSk9owHUUShR4LiF8ZBh0TYoCBwpzjA8PU1VF4yWUk%2FkKhcHE1AnhVnx9Ok97aXjgDlcy%2FBJBOGOrAWlDTVjfcUuMOY2ZXjMByX8xBZNvz0QlJhGrImFAwnDuVREe5UHPZn0UKC0ASbtq4WkNcQGNCPrmpmU3O9Ns47Bz64nTzKgdjmN7X77RgMKI0TVcWou2XO%2FcNmMhQaxaomrt0skbdomNXKLry508WLjoUufbyBGjBniPK%2B1XrhngvyWtSQ%3D%3D&X-Amz-Signature=28c61894ae8a475b26d450f410f16acd2d93ca6759f2767187f5aec975dcca9b" -o indexes/faiss.index && \
    echo "✅ Downloaded FAISS index (160MB)" && \
    \
    echo "📥 Downloading chunks metadata (largest file last)..." && \
    curl -L "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes/chunks_metadata.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIASVQKHXMDX32Z6GYL%2F20250801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250801T130451Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIA93iDQUP7gOU1mn%2B6ve2Tt1nfElvl0g3yzSQmAb5APyAiEAztEGSPvTpdodil%2FqTNfHwnWaj%2BV3gRLIvwHr%2FRkIRhsqgQMI7v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgwxODM2MzEzMzAwNTUiDM9KnkoaW061F46cDCrVAng7NpVXswPWitWSucJWsEkSRY0nq96DZgwx0TGzMnzCYrZiZ0HUcs7t3ytsWB3%2BpjQMOKdzHDCnUkdngUt89QiIQxlo3eLleAXBl9sfafvIkmBXh4cDs3T1OJs44Dh%2BoOpkMfm1XM5bkBcoic78oHsHkST4mq2KSC9SMZc%2Ffza865sgro0c7Fw6TvqvsVaqmma7i368zk%2FQV80ciLtO2nPUGGdbSvcVy3MxncDzoafFl71yNQ8KMNUybECai8Lw%2Fvz8FJeQlu4gKPTA%2F8k%2FKGi8vg7EK%2Byp3OOAJ4rTPwTEdQfAdU5ULuYtUBYKnVytvKEIZJzi01l4L9f%2BMNksst3F2WvMdXtKhJQJgUPsgpRem0GP04zTuxid2ZCX0OUh7LpKZ3NwLMFe0yt1mvGZducIFwFzcHZMPQmTxusJtlY2P55rHTVvhr8izSIqpkE0ZOmfvPqqMIjxssQGOq0C5rs6RBzaq0yalBHnbW3rzzXUkNFSM%2Bezx%2BudbJ%2BzPtjTu49rHWrb%2FuaRB%2FBAx%2BEE1XpacHsPsrYSAqPuvTD7IoO2Rt4qBq4UcszJ7U3UguTdRmcWKvLZ43viw%2Fc%2BMmOwGQuU%2BAIeMDOfYw58r3ArSk9owHUUShR4LiF8ZBh0TYoCBwpzjA8PU1VF4yWUk%2FkKhcHE1AnhVnx9Ok97aXjgDlcy%2FBJBOGOrAWlDTVjfcUuMOY2ZXjMByX8xBZNvz0QlJhGrImFAwnDuVREe5UHPZn0UKC0ASbtq4WkNcQGNCPrmpmU3O9Ns47Bz64nTzKgdjmN7X77RgMKI0TVcWou2XO%2FcNmMhQaxaomrt0skbdomNXKLry508WLjoUufbyBGjBniPK%2B1XrhngvyWtSQ%3D%3D&X-Amz-Signature=5865d4a20a5bf8f75b3d0c1ba616222f517c25e76e29eaca859ba7fbf661ddd7" -o indexes/chunks_metadata.json && \
    echo "✅ Downloaded chunks metadata (1.3GB)"

# Only copy code AFTER successful downloads
COPY . .

# Verify files were downloaded successfully
RUN ls -la indexes/ && \
    echo "Downloaded files:" && \
    du -h indexes/*

# Use PORT environment variable (works for Railway, Cloud Run, etc.)
CMD uvicorn fastapi_backend:app --host 0.0.0.0 --port $PORT
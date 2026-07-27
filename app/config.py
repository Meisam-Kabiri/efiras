"""
Application configuration and constants.

This module centralizes all configuration values, magic numbers, and settings
used throughout the EFIRAS application.
"""

import os
from typing import List

# API Configuration
API_TITLE = "EFIRAS API"
API_DESCRIPTION = "Regulatory AI Assistant with Authentication"
API_VERSION = "1.0.0"

# Server Configuration
DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"
REQUEST_TIMEOUT_SECONDS = 300

# CORS Configuration
ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOW_CREDENTIALS = True
ALLOWED_METHODS = ["GET", "POST", "DELETE", "OPTIONS"]
ALLOWED_HEADERS = ["*"]

# Search Configuration
DEFAULT_TOP_K_PUBLIC = 12
DEFAULT_TOP_K_AUTHENTICATED = 15

# Rate Limiting
RATE_LIMIT_ENABLED = True

# File Paths & Index Configuration
INDEX_DIR = os.getenv("INDEX_DIR", "data/regulatory_indexes")
FAISS_FILENAME = os.getenv("FAISS_FILENAME", "regulatory_faiss.bin")
DB_FILENAME = os.getenv("DB_FILENAME", "regulatory_chunks.db")
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOGS_DIR = "logs"

# Remote Resources
BASE_URL = "https://storage.googleapis.com/efiras-faiss-index/indexes"
REQUIRED_INDEX_FILES = [
    {
        "filename": FAISS_FILENAME,
        "path": os.path.join(INDEX_DIR, FAISS_FILENAME),
        "url": f"{BASE_URL}/{FAISS_FILENAME}",
    },
    {
        "filename": DB_FILENAME,
        "path": os.path.join(INDEX_DIR, DB_FILENAME),
        "url": f"{BASE_URL}/{DB_FILENAME}",
    },
]

# Chunk Size Configuration
DEFAULT_CHUNK_SIZE = 1500
MAX_QUERY_LENGTH = 10000

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

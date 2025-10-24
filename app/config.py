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
# TODO: Replace with specific origins for production
ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOW_CREDENTIALS = True
ALLOWED_METHODS = ["GET", "POST", "DELETE", "OPTIONS"]
ALLOWED_HEADERS = ["*"]

# Search Configuration
DEFAULT_TOP_K_PUBLIC = 12
DEFAULT_TOP_K_AUTHENTICATED = 15

# Rate Limiting
RATE_LIMIT_ENABLED = True

# File Paths
INDEX_DIR = "data/indexes"
LOGS_DIR = "logs"

# Remote Resources
BASE_URL = "https://efiras-indexes.s3.us-east-1.amazonaws.com/indexes"
REQUIRED_INDEX_FILES = [
    {
        "url": f"{BASE_URL}/bm25_tokenized.pkl",
        "path": f"{INDEX_DIR}/bm25_tokenized.pkl",
    },
    {"url": f"{BASE_URL}/faiss.index", "path": f"{INDEX_DIR}/faiss.index"},
    {
        "url": f"{BASE_URL}/chunks_metadata.json",
        "path": f"{INDEX_DIR}/chunks_metadata.json",
    },
]

# Chunk Size Configuration
DEFAULT_CHUNK_SIZE = 1500
MAX_QUERY_LENGTH = 1000

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

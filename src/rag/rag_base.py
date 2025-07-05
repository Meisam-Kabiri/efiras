# rag_base.py
"""
Base classes and common utilities for RAG pipelines.
"""

from abc import ABC, abstractmethod
from typing import List

class BaseRAG(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> None:
        pass

    # @abstractmethod
    # def retrieve(self, query: str) -> List[str]:
    #     pass

    @abstractmethod
    def generate_answer(self, query: str) -> str:
        pass

# utils.py
from openai import OpenAI
import os


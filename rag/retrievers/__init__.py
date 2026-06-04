"""Retriever implementations."""

from rag.retrievers.base import Retriever
from rag.retrievers.dense import DenseRetriever

__all__ = [
    "DenseRetriever",
    "Retriever",
]

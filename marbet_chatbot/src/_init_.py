"""
Marbet Event Assistant Chatbot package.

This package contains modules for document processing, RAG pipelines,
and chatbot functionality for the Marbet event assistant.
"""

__version__ = "1.0.0"

from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline
from src.chatbot import MarbetChatbot

__all__ = ["DocumentProcessor", "RAGPipeline", "MarbetChatbot"]
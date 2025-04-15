"""
Main chatbot implementation for the Marbet Event Assistant.

This module provides the main MarbetChatbot class that integrates
document processing and RAG pipeline functionality.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline

# Set up logging
logger = logging.getLogger("marbet_chatbot.chatbot")


class MarbetChatbot:
    """Main chatbot class for the Marbet event assistant"""

    def __init__(self,
                 ollama_url: str = "http://194.171.191.226:3061",
                 model_name: str = "llama3.1:8b",
                 docs_dir: str = "./docs",
                 db_dir: str = "./chroma_db"):
        """
        Initialize the chatbot

        Args:
            ollama_url: URL of the Ollama server
            model_name: Name of the model to use
            docs_dir: Directory containing documents
            db_dir: Directory for vector store
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.docs_dir = docs_dir
        self.db_dir = db_dir

        # Components
        self.document_processor = None
        self.pipeline = None

        # Chat history
        self.chat_history = []

        # Initialization status
        self.is_initialized = False

    def initialize(self, rebuild_db: bool = False) -> 'MarbetChatbot':
        """
        Initialize the chatbot with document processing and RAG setup

        Args:
            rebuild_db: Whether to rebuild the vector database

        Returns:
            Self for method chaining
        """
        logger.info("Initializing Marbet Chatbot")

        # Initialize document processor
        self.document_processor = DocumentProcessor(document_dir=self.docs_dir)

        # Initialize RAG pipeline
        self.pipeline = RAGPipeline(
            ollama_url=self.ollama_url,
            model_name=self.model_name
        )

        # Check if we need to build or rebuild the vector database
        if rebuild_db or not os.path.exists(self.db_dir):
            logger.info("Building new vector database")

            # Process documents
            processed_data = self.document_processor.process_all()

            # Initialize pipeline with processed documents
            self.pipeline.initialize_pipeline(
                document_chunks=processed_data["chunks"],
                persist_dir=self.db_dir
            )
        else:
            logger.info("Loading existing vector database")
            self.pipeline.initialize_pipeline(persist_dir=self.db_dir)

        self.is_initialized = True
        logger.info("Chatbot initialization complete")
        return self

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return response

        Args:
            query: User query text

        Returns:
            Dictionary with query, response, and status
        """
        if not self.is_initialized:
            raise ValueError("Chatbot not initialized. Call initialize() first.")

        logger.info(f"Processing user query: {query}")

        # Add query to chat history
        self.chat_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })

        # Get response from RAG pipeline
        try:
            response = self.pipeline.query(query)

            # Add response to chat history
            self.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "query": query,
                "response": response,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = "I'm sorry, I encountered an error while processing your request. Please try again."

            # Add error response to chat history
            self.chat_history.append({
                "role": "assistant",
                "content": error_response,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "query": query,
                "response": error_response,
                "success": False,
                "error": str(e)
            }

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Any]:
        """
        Get relevant documents for a query

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of relevant document objects
        """
        if not self.is_initialized:
            raise ValueError("Chatbot not initialized. Call initialize() first.")

        logger.info(f"Getting relevant documents for query: {query}")
        return self.pipeline.get_relevant_documents(query, k=k)

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Return the chat history

        Returns:
            List of chat messages with roles and content
        """
        return self.chat_history

    def clear_chat_history(self) -> None:
        """Clear the chat history"""
        self.chat_history = []
        logger.info("Chat history cleared")


if __name__ == "__main__":
    # Setup basic logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create and initialize chatbot
    chatbot = MarbetChatbot(
        ollama_url="http://194.171.191.226:3061",
        model_name="llama3.1:8b",
        docs_dir="./docs",
        db_dir="./chroma_db"
    ).initialize()

    # Test query
    query = "What activities are available in Halifax?"
    result = chatbot.process_query(query)

    print(f"\nQuery: {result['query']}")
    print(f"Response: {result['response']}")
"""
Utility functions for the Marbet Event Assistant Chatbot.
"""

import os
import logging
import sys
from typing import Dict, Any, Optional

def setup_logging(log_file: str = "chatbot.log", console_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_file: Path to the log file
        console_level: Logging level for console output

    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("marbet_chatbot")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def ensure_directories(directories: list) -> None:
    """
    Ensure that the specified directories exist.

    Args:
        directories: List of directory paths to create if they don't exist
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def format_document_chunk(doc) -> Dict[str, Any]:
    """
    Format a document chunk for display.

    Args:
        doc: Document chunk object

    Returns:
        Dictionary with formatted document information
    """
    return {
        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
        "metadata": {
            "filename": doc.metadata.get("filename", "Unknown"),
            "category": doc.metadata.get("category", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "chunk_id": doc.metadata.get("chunk_id", "Unknown")
        }
    }

def print_welcome_message() -> None:
    """
    Print the welcome message for the CLI application.
    """
    print("\n" + "="*80)
    print("                  MARBET EVENT ASSISTANT - SALES TRIP 2024")
    print("="*80)
    print("\nWelcome to the Marbet Event Assistant!")
    print("This AI assistant can help you with information about:")
    print("  • Activities and excursions for each destination")
    print("  • Ship services and amenities")
    print("  • Travel requirements and documents")
    print("  • Visa requirements (ESTA for USA, eTA for Canada)")
    print("  • General travel assistance")
    print("\nType your questions or 'exit' to quit.")
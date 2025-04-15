"""
Document processing module for the Marbet Event Assistant Chatbot.

This module handles loading, processing, and chunking of documents for the RAG system.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Set up logging
logger = logging.getLogger("marbet_chatbot.document_processor")


class DocumentProcessor:
    """Process documents for the RAG system"""

    def __init__(self, document_dir: str = "./docs"):
        """
        Initialize the document processor

        Args:
            document_dir: Directory containing document files
        """
        self.document_dir = document_dir

    def process_all(self) -> Dict[str, Any]:
        """
        Process all documents in the document directory

        Returns:
            Dictionary containing original documents and processed chunks
        """
        # Load raw documents
        documents = self._load_documents()

        # Extract metadata and enhance documents
        enhanced_docs = self._enhance_documents(documents)

        # Split documents into chunks
        chunks = self._split_documents(enhanced_docs)

        return {
            "documents": enhanced_docs,
            "chunks": chunks
        }

    def _load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the document directory

        Returns:
            List of document objects
        """
        logger.info(f"Loading documents from {self.document_dir}")

        documents = []

        # Check if directory exists
        if not os.path.exists(self.document_dir):
            logger.error(f"Document directory {self.document_dir} does not exist")
            raise FileNotFoundError(f"Document directory {self.document_dir} does not exist")

        # Load each PDF file
        for filename in os.listdir(self.document_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.document_dir, filename)

                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} pages from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

        logger.info(f"Loaded {len(documents)} document pages in total")
        return documents

    def _enhance_documents(self, documents: List[Document]) -> List[Document]:
        """
        Enhance documents with additional metadata

        Args:
            documents: List of document objects

        Returns:
            Enhanced document objects
        """
        enhanced_docs = []

        for doc in documents:
            # Get filename from source path
            filename = os.path.basename(doc.metadata.get("source", "unknown"))

            # Add filename to metadata
            doc.metadata["filename"] = filename

            # Categorize document
            doc.metadata["category"] = self._categorize_document(filename)

            # Extract dates if applicable
            if doc.metadata["category"] == "activities":
                dates = self._extract_dates(doc.page_content)
                if dates:
                    doc.metadata["dates"] = dates

            # Extract locations if applicable
            if doc.metadata["category"] == "activities":
                locations = self._extract_locations(doc.page_content)
                if locations:
                    doc.metadata["locations"] = locations

            enhanced_docs.append(doc)

        logger.info(f"Enhanced {len(enhanced_docs)} documents with metadata")
        return enhanced_docs

    def _categorize_document(self, filename: str) -> str:
        """
        Categorize document based on filename

        Args:
            filename: Name of the document file

        Returns:
            Category string
        """
        categories = {
            "Activity": "activities",
            "WiFi": "technology",
            "A-Z": "ship_services",
            "Eclipse": "ship_services",
            "Pack": "travel_requirements",
            "SPA": "wellness",
            "ESTA": "visa_requirements",
            "eTA": "visa_requirements"
        }

        for key, category in categories.items():
            if key in filename:
                return category

        return "general"

    def _extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text

        Args:
            text: Document text

        Returns:
            List of date strings
        """
        # Pattern for dates in format DD.MM.YYYY
        date_pattern = r'\d{2}\.\d{2}\.202\d'
        dates = re.findall(date_pattern, text)

        # Also look for dates with day names
        day_patterns = [
            r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+\d{2}\.\d{2}\.202\d',
            r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+202\d'
        ]

        for pattern in day_patterns:
            dates.extend(re.findall(pattern, text))

        return list(set(dates))  # Remove duplicates

    def _extract_locations(self, text: str) -> List[str]:
        """
        Extract locations from text

        Args:
            text: Document text

        Returns:
            List of location strings
        """
        # Look for location patterns in activities document
        locations = []

        location_patterns = [
            r'Halifax',
            r'Lunenburg',
            r'Portland',
            r'Boston',
            r'Provincetown',
            r'Marthas Vineyard',
            r'Martha\'s Vineyard',
            r'New York',
            r'New York City'
        ]

        for location in location_patterns:
            if re.search(location, text):
                locations.append(location)

        return list(set(locations))  # Remove duplicates

    def _split_documents(self, documents: List[Document],
                         chunk_size: int = 1000,
                         chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval

        Args:
            documents: List of document objects
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            List of document chunks
        """
        logger.info(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Split documents
        chunks = text_splitter.split_documents(documents)

        # Enhance chunks with additional metadata
        for i, chunk in enumerate(chunks):
            # Add chunk ID
            chunk.metadata["chunk_id"] = i

            # Preserve original metadata
            # Everything from the original document is already copied over

        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks


if __name__ == "__main__":
    # Setup basic logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Test the document processor
    processor = DocumentProcessor(document_dir="./docs")
    result = processor.process_all()

    print(f"Processed {len(result['documents'])} documents into {len(result['chunks'])} chunks")

    # Display sample chunks
    for i, chunk in enumerate(result['chunks'][:3]):
        print(f"\nChunk {i + 1}:")
        print(f"Category: {chunk.metadata.get('category')}")
        print(f"Filename: {chunk.metadata.get('filename')}")
        print(f"Excerpt: {chunk.page_content[:150]}...")
"""
RAG pipeline implementation for the Marbet Event Assistant Chatbot.

This module implements the Retrieval-Augmented Generation (RAG) pipeline
for the chatbot, handling embeddings, vector storage, and query processing.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logger = logging.getLogger("marbet_chatbot.rag_pipeline")


class RAGPipeline:
    """RAG Pipeline implementation for the chatbot"""

    def __init__(self,
                 ollama_url: str = "http://194.171.191.226:3061",
                 model_name: str = "llama3.1:8b"):
        """
        Initialize the RAG pipeline

        Args:
            ollama_url: URL of the Ollama server
            model_name: Name of the model to use
        """
        self.ollama_url = ollama_url
        self.model_name = model_name

        # Components to be initialized
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.prompt = None
        self.chain = None
        self.memory = None

        # System prompt for the chatbot
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the chatbot"""
        return """
        # Marbet AI Event Assistant

        ## Identity
        You are an AI event assistant for Marbet's incentive trip aboard the Scenic Eclipse ship, 
        traveling to Canada and the USA from October 4-11, 2024. Your purpose is to provide 
        travelers with accurate and helpful information about their journey.

        ## Core Information Categories
        - Activities and excursions for each destination
        - Ship services and amenities
        - Travel requirements and documents
        - WiFi and technology
        - Visa requirements (ESTA for USA, eTA for Canada)
        - General travel assistance

        ## Tone and Style
        - Professional and courteous
        - Clear and concise
        - Practical and informative
        - Service-oriented

        ## Response Guidelines
        1. Answer questions based ONLY on information in the provided context
        2. When information is available, be specific and detailed
        3. If information isn't in the context, politely acknowledge and offer to help with something else
        4. For schedule-related questions, include specific dates and times when available
        5. Format your responses for easy reading, using bullet points when appropriate
        6. Highlight important information that travelers need to know
        7. When referencing documents, you can mention the source (e.g., "According to the Activity Overview...")

        Remember, your goal is to help travelers have a smooth, enjoyable experience by providing 
        accurate information and practical assistance.
        """

    def initialize_pipeline(self,
                            document_chunks: Optional[List[Document]] = None,
                            persist_dir: str = "./chroma_db") -> 'RAGPipeline':
        """
        Initialize the RAG pipeline

        Args:
            document_chunks: Document chunks to create vector store (if provided)
            persist_dir: Directory for vector store persistence

        Returns:
            Self for method chaining
        """
        # Initialize embeddings
        self._initialize_embeddings()

        # Initialize vector store
        if document_chunks is not None:
            self._create_vectorstore(document_chunks, persist_dir)
        else:
            self._load_vectorstore(persist_dir)

        # Initialize retriever
        self._initialize_retriever()

        # Initialize LLM
        self._initialize_llm()

        # Initialize prompt
        self._initialize_prompt()

        # Initialize memory
        self._initialize_memory()

        # Initialize chain
        self._initialize_chain()

        logger.info("RAG pipeline initialized successfully")
        return self

    def _initialize_embeddings(self):
        """Initialize embeddings model"""
        logger.info(f"Initializing embeddings with model {self.model_name}")

        self.embeddings = OllamaEmbeddings(
            base_url=self.ollama_url,
            model=self.model_name
        )

    def _create_vectorstore(self, document_chunks: List[Document], persist_dir: str):
        """
        Create and persist vector store

        Args:
            document_chunks: Document chunks for vector store
            persist_dir: Directory for vector store persistence
        """
        logger.info(f"Creating vector store in {persist_dir} with {len(document_chunks)} chunks")

        # Create directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)

        # Create and persist vector store
        self.vectorstore = Chroma.from_documents(
            documents=document_chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )

        # Persist to disk
        self.vectorstore.persist()

        logger.info(f"Vector store created and persisted to {persist_dir}")

    def _load_vectorstore(self, persist_dir: str):
        """
        Load existing vector store from disk

        Args:
            persist_dir: Directory where vector store is persisted
        """
        logger.info(f"Loading vector store from {persist_dir}")

        # Check if directory exists
        if not os.path.exists(persist_dir):
            logger.error(f"Vector store directory {persist_dir} does not exist")
            raise FileNotFoundError(f"Vector store directory {persist_dir} does not exist")

        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

        logger.info(f"Vector store loaded from {persist_dir}")

    def _initialize_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize retriever component

        Args:
            search_kwargs: Search parameters for retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vector store must be initialized first")

        if search_kwargs is None:
            search_kwargs = {
                "k": 5  # Number of documents to retrieve
            }

        logger.info(f"Initializing retriever with parameters: {search_kwargs}")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def _initialize_llm(self, temperature: float = 0.1):
        """
        Initialize LLM component

        Args:
            temperature: Temperature for LLM generation
        """
        logger.info(f"Initializing LLM with model {self.model_name} and temperature {temperature}")

        self.llm = ChatOllama(
            base_url=self.ollama_url,
            model=self.model_name,
            temperature=temperature
        )

    def _initialize_prompt(self):
        """Initialize prompt template"""
        # Create a chat prompt template that works better with modern LangChain
        template = """
        You are an AI event assistant for Marbet, a German event management agency. You are helping travelers with their incentive trip aboard the Scenic Eclipse ship traveling to Canada and the USA from October 4-11, 2024.

        Your role is to provide accurate information about the trip itinerary, activities, ship services, and travel requirements. Answer questions clearly and concisely based only on the information provided in the context.

        Context Information:
        {context}

        Question: {query}

        Guidelines:
        1. Answer based ONLY on the information in the context
        2. If you don't have enough information, politely say so
        3. Be specific about dates, locations, and times when available
        4. Format your response for easy reading
        5. When multiple options exist, present them clearly

        Please provide a helpful answer:
        """

        self.prompt = PromptTemplate.from_template(template)
        logger.info("Prompt template initialized")

    def _initialize_memory(self):
        """Initialize conversation memory"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        logger.info("Conversation memory initialized")

    def _initialize_chain(self):
        """Initialize RAG chain"""
        if not all([self.retriever, self.llm, self.prompt]):
            raise ValueError("Retriever, LLM, and prompt must be initialized first")

        logger.info("Initializing RAG chain")

        # Create a modern LangChain RAG pipeline using the LCEL (LangChain Expression Language)
        # This approach is more reliable with newer versions of LangChain

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Build the RAG chain
        self.chain = (
                {"context": self.retriever | format_docs, "query": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

        logger.info("RAG chain initialized")

    def query(self, question: str) -> str:
        """
        Process a query through the RAG chain

        Args:
            question: User question

        Returns:
            Response text
        """
        if self.chain is None:
            raise ValueError("Chain not initialized. Call initialize_pipeline() first.")

        logger.info(f"Processing query: {question}")

        try:
            # With the new chain structure, we can just invoke with the query
            response = self.chain.invoke(question)
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I'm sorry, I encountered an error while processing your request. Please try again."

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Get relevant documents for a query without generating a response

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call initialize_pipeline() first.")

        return self.retriever.get_relevant_documents(query)


if __name__ == "__main__":
    # Setup basic logging for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Test with existing vector store
    try:
        pipeline = RAGPipeline(
            ollama_url="http://194.171.191.226:3061",
            model_name="llama3.1:8b"
        )

        pipeline.initialize_pipeline(persist_dir="./chroma_db")

        # Test query
        query = "What activities are available in Halifax?"
        response = pipeline.query(query)

        print(f"\nQuery: {query}")
        print(f"Response: {response}")

    except FileNotFoundError:
        print("Vector store does not exist. Please create it first with document chunks.")
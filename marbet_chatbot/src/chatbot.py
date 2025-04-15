import argparse
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import our custom modules (assuming they're in the same directory)
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MarbetChatbot:
    """Main chatbot class for the Marbet event assistant"""

    def __init__(self,
                 ollama_url: str = "http://194.171.191.226:3061",
                 model_name: str = "llama3.1:8b",
                 docs_dir: str = "./docs",
                 db_dir: str = "./chroma_db"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.pipeline = None
        self.chat_history = []

        # Initialize the system prompt
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

    def initialize(self, rebuild_db: bool = False):
        """Initialize the chatbot with document processing and RAG setup"""
        logger.info("Initializing Marbet Chatbot")

        # Initialize RAG pipeline
        self.pipeline = RAGPipeline(
            ollama_url=self.ollama_url,
            model_name=self.model_name
        )

        # Check if we need to build or rebuild the vector database
        if rebuild_db or not os.path.exists(self.db_dir):
            logger.info("Building new vector database")

            # Process documents
            doc_processor = DocumentProcessor(document_dir=self.docs_dir)
            processed_data = doc_processor.process_all()

            # Initialize pipeline with processed documents
            self.pipeline.initialize_pipeline(
                document_chunks=processed_data["chunks"],
                persist_dir=self.db_dir
            )
        else:
            logger.info("Loading existing vector database")
            self.pipeline.initialize_pipeline(persist_dir=self.db_dir)

        logger.info("Chatbot initialization complete")
        return self

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return response"""
        if not self.pipeline:
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

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Return the chat history"""
        return self.chat_history

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
        logger.info("Chat history cleared")


def run_cli():
    """Run a command-line interface for the chatbot"""
    parser = argparse.ArgumentParser(description="Marbet Event Assistant Chatbot")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector database")
    parser.add_argument("--docs-dir", type=str, default="./docs", help="Directory containing documents")
    parser.add_argument("--db-dir", type=str, default="./chroma_db", help="Directory for vector database")
    parser.add_argument("--ollama-url", type=str, default="http://194.171.191.226:3061", help="Ollama server URL")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model name to use")

    args = parser.parse_args()

    # Print welcome message
    print("\n" + "=" * 80)
    print("                    MARBET EVENT ASSISTANT - SALES TRIP 2024")
    print("=" * 80)
    print("\nInitializing AI assistant. This may take a moment...\n")

    # Initialize chatbot
    chatbot = MarbetChatbot(
        ollama_url=args.ollama_url,
        model_name=args.model,
        docs_dir=args.docs_dir,
        db_dir=args.db_dir
    ).initialize(rebuild_db=args.rebuild)

    print("\nMarbet Event Assistant is ready to help with your trip questions!")
    print("Ask about activities, ship services, travel requirements, or anything else related to your trip.")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.\n")

    # Main interaction loop
    while True:
        try:
            user_input = input("You: ")

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("\nThank you for using the Marbet Event Assistant. Have a wonderful trip!")
                break

            # Process query
            result = chatbot.process_query(user_input)
            print(f"\nAssistant: {result['response']}\n")

        except KeyboardInterrupt:
            print("\n\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again or restart the application.")


if __name__ == "__main__":
    run_cli()
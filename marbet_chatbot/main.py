#!/usr/bin/env python
"""
Main entry point for the Marbet Event Assistant Chatbot.

This script provides command-line and API interfaces to interact with the chatbot.
"""

import os
import argparse
import sys
from typing import Dict, Any, List, Optional

# Import from src package
from src.chatbot import MarbetChatbot
from src.utils import setup_logging, ensure_directories, print_welcome_message

# Set up logging
logger = setup_logging()


def run_cli():
    """Run the chatbot in command-line interface mode"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Marbet Event Assistant Chatbot")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector database")
    parser.add_argument("--docs-dir", type=str, default="./docs", help="Directory containing documents")
    parser.add_argument("--db-dir", type=str, default="./chroma_db", help="Directory for vector database")
    parser.add_argument("--ollama-url", type=str, default="http://194.171.191.226:3061", help="Ollama server URL")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model name to use")

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories([args.docs_dir, args.db_dir])

    # Print welcome message
    print_welcome_message()
    print("Initializing AI assistant. This may take a moment...\n")

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
            logger.error(f"Error in CLI mode: {e}")
            print(f"\nAn error occurred: {e}")
            print("Please try again or restart the application.")


def run_api():
    """Run the chatbot as a REST API server"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        logger.error("FastAPI or uvicorn not installed. Run: pip install fastapi uvicorn")
        print("FastAPI or uvicorn not installed. Run: pip install fastapi uvicorn")
        return

    # Parse arguments
    parser = argparse.ArgumentParser(description="Marbet Event Assistant API")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector database")
    parser.add_argument("--docs-dir", type=str, default="./docs", help="Directory containing documents")
    parser.add_argument("--db-dir", type=str, default="./chroma_db", help="Directory for vector database")
    parser.add_argument("--ollama-url", type=str, default="http://194.171.191.226:3061", help="Ollama server URL")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model name to use")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories([args.docs_dir, args.db_dir])

    # Initialize chatbot
    print("Initializing Marbet Event Assistant. This may take a moment...")
    chatbot = MarbetChatbot(
        ollama_url=args.ollama_url,
        model_name=args.model,
        docs_dir=args.docs_dir,
        db_dir=args.db_dir
    ).initialize(rebuild_db=args.rebuild)

    # Define API models
    class QueryRequest(BaseModel):
        query: str

    class QueryResponse(BaseModel):
        query: str
        response: str
        success: bool
        error: Optional[str] = None

    # Create FastAPI app
    app = FastAPI(
        title="Marbet Event Assistant API",
        description="API for accessing the Marbet Event Assistant chatbot",
        version="1.0.0"
    )

    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Process a query to the chatbot"""
        try:
            result = chatbot.process_query(request.query)
            return result
        except Exception as e:
            logger.error(f"API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/history")
    async def get_history():
        """Get the chat history"""
        return {"history": chatbot.get_chat_history()}

    @app.post("/api/clear-history")
    async def clear_history():
        """Clear the chat history"""
        chatbot.clear_chat_history()
        return {"message": "Chat history cleared"}

    # Run the API server
    print(f"Starting API server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    # Parse mode argument
    parser = argparse.ArgumentParser(description="Marbet Event Assistant")
    parser.add_argument("--mode", type=str, choices=["cli", "api"], default="cli",
                        help="Run mode: command line interface or API server")

    # Parse just the mode argument first
    args, remaining = parser.parse_known_args()

    # Set remaining args for the next parser
    sys.argv = [sys.argv[0]] + remaining

    # Run in the selected mode
    if args.mode == "cli":
        run_cli()
    elif args.mode == "api":
        run_api()
    else:
        print(f"Unknown mode: {args.mode}")
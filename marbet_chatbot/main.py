"""
Marbet Event Assistant Chatbot
Project main entry point
"""

import os
import argparse
import logging
from chatbot import MarbetChatbot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Marbet Event Assistant Chatbot")
    parser.add_argument("--mode", type=str, choices=["cli", "api"], default="cli",
                        help="Run mode: command line interface or API server")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the vector database")
    parser.add_argument("--docs-dir", type=str, default="./docs",
                        help="Directory containing documents")
    parser.add_argument("--db-dir", type=str, default="./chroma_db",
                        help="Directory for vector database")
    parser.add_argument("--ollama-url", type=str, default="http://194.171.191.226:3061",
                        help="Ollama server URL")
    parser.add_argument("--model", type=str, default="llama3.1:8b",
                        help="Model name to use")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host for API server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for API server")

    args = parser.parse_args()

    # Create necessary directories if they don't exist
    os.makedirs(args.docs_dir, exist_ok=True)
    os.makedirs(args.db_dir, exist_ok=True)

    # Initialize chatbot
    chatbot = MarbetChatbot(
        ollama_url=args.ollama_url,
        model_name=args.model,
        docs_dir=args.docs_dir,
        db_dir=args.db_dir
    )

    # Initialize and run in the selected mode
    if args.mode == "cli":
        run_cli_mode(chatbot, args)
    elif args.mode == "api":
        run_api_mode(chatbot, args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        print(f"Unknown mode: {args.mode}")


def run_cli_mode(chatbot, args):
    """Run the chatbot in CLI mode"""
    logger.info("Starting chatbot in CLI mode")

    # Initialize the chatbot
    chatbot.initialize(rebuild_db=args.rebuild)

    # Print welcome message
    print("\n" + "=" * 80)
    print("                    MARBET EVENT ASSISTANT - SALES TRIP 2024")
    print("=" * 80)
    print("\nWelcome to the Marbet Event Assistant!")
    print("I can help you with information about your trip itinerary, activities,")
    print("ship services, travel requirements, and more.")
    print("\nHow can I assist you today?")
    print("(Type 'exit', 'quit', or 'bye' to end the conversation)")

    # Main interaction loop
    while True:
        try:
            user_input = input("\nYou: ")

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("\nThank you for using the Marbet Event Assistant. Have a wonderful trip!")
                break

            # Process query
            result = chatbot.process_query(user_input)
            print(f"\nAssistant: {result['response']}")

        except KeyboardInterrupt:
            print("\n\nSession terminated by user.")
            break
        except Exception as e:
            logger.error(f"Error in CLI mode: {e}")
            print(f"\nAn error occurred: {e}")
            print("Please try again or restart the application.")


def run_api_mode(chatbot, args):
    """Run the chatbot as an API server"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        logger.error("FastAPI or uvicorn not installed. Run: pip install fastapi uvicorn")
        print("FastAPI or uvicorn not installed. Run: pip install fastapi uvicorn")
        return

    logger.info("Starting chatbot in API mode")

    # Initialize the chatbot
    chatbot.initialize(rebuild_db=args.rebuild)

    # Create API app
    app = FastAPI(title="Marbet Event Assistant API",
                  description="API for the Marbet Event Assistant Chatbot",
                  version="1.0.0")

    class QueryRequest(BaseModel):
        query: str

    class QueryResponse(BaseModel):
        query: str
        response: str
        success: bool
        error: str = None

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
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
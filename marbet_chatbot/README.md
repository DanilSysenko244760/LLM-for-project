# Marbet Event Assistant Chatbot

A RAG-enhanced chatbot providing event assistance based on Marbet's documents. The chatbot is built using Python, LangChain, and Ollama, demonstrating the ability to retrieve and deliver event-specific information in an accurate and helpful manner.

## Project Structure

```
marbet_chatbot/
├── docs/                      # Document directory
│   ├── Activity_Overview.pdf
│   ├── Guest_WiFi.pdf
│   ├── Information_A-Z_Scenic_Eclipse_I.pdf
│   ├── Packlist.pdf
│   ├── SPA_brochure.pdf
│   ├── Tutorial_ESTA.pdf
│   └── Tutorial_eTA.pdf
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── document_processor.py  # Document processing
│   ├── rag_pipeline.py        # RAG implementation
│   ├── chatbot.py             # Main chatbot logic
│   └── utils.py               # Helper functions
│
├── chroma_db/                 # Vector store persistence
│
├── tests/                     # Test directory
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Key Features

- **Document Processing**: Automatically loads, processes, and chunks PDF documents
- **Metadata Extraction**: Extracts metadata like dates and locations from documents
- **RAG Implementation**: Uses Retrieval-Augmented Generation for accurate responses
- **Vector Storage**: Efficiently stores and retrieves document embeddings
- **Multiple Interfaces**: Supports both CLI and API interfaces

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/marbet-chatbot.git
   cd marbet-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the necessary models:
   ```bash
   # Ensure Ollama is running
   ollama pull llama3.1:8b
   ```

## Usage

### Command Line Interface

```bash
# Run with default settings
python main.py

# Rebuild the vector database
python main.py --rebuild

# Specify a different model
python main.py --model llama3.2
```

### API Server

```bash
# Run the API server
python main.py --mode api

# Specify host and port
python main.py --mode api --host 127.0.0.1 --port 8080
```

## API Endpoints

- **POST /api/query**: Process a query
  ```json
  {
    "query": "What activities are available in Halifax?"
  }
  ```

- **GET /api/history**: Get chat history

- **POST /api/clear-history**: Clear chat history

## Document Categories

The chatbot recognizes and categorizes documents based on their content:

- **Activities**: Information about tours, excursions, and leisure activities
- **Ship Services**: Onboard amenities, policies, and services
- **Travel Requirements**: Packing lists, documents, etc.
- **Visa Requirements**: ESTA (USA) and eTA (Canada) information
- **Technology**: WiFi and other tech-related information
- **Wellness**: Spa services and wellness options

## Configuration Options

- `--docs-dir`: Directory containing documents (default: "./docs")
- `--db-dir`: Directory for vector store (default: "./chroma_db")
- `--ollama-url`: Ollama server URL (default: "http://194.171.191.226:3061")
- `--model`: Model name to use (default: "llama3.1:8b")
- `--rebuild`: Force rebuilding the vector database

## Development

### Running Individual Components

Each module can be run independently for testing:

```bash
# Test document processor
python -m src.document_processor

# Test RAG pipeline
python -m src.rag_pipeline

# Test chatbot
python -m src.chatbot
```

### Adding New Documents

1. Place new PDF documents in the `docs/` directory
2. Run the chatbot with the `--rebuild` flag to include the new documents:
   ```bash
   python main.py --rebuild
   ```

## License

[Insert License Information]

## Contributors

[Insert Contributor Information]
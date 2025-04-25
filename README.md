# Marbet Event Assistant

A Retrieval-Augmented Generation (RAG) chatbot developed for Marbet to assist travelers during their incentive trip aboard the Scenic Eclipse ship. This AI assistant provides information about activities, ship services, travel requirements, and more.

## Features

- **Document Analysis**: Processes text and images from PDF documents
- **Intelligent Retrieval**: Finds relevant information from trip documents
- **Conversational Interface**: Natural language interactions for trip questions
- **Multi-Interface Support**: CLI and API options for different use cases
- **Sources Tracking**: Shows information sources for full transparency

## Project Structure

```
marbet_assistant/
├── data/                       # Document directory
│   ├── Activity_Overview.pdf
│   ├── Guest_WiFi.pdf
│   ├── Information_A-Z_Scenic_Eclipse_I.pdf 
│   ├── Packlist.pdf
│   ├── SPA_brochure.pdf
│   ├── Tutorial_ESTA.pdf
│   └── Tutorial_eTA.pdf
│
├── src/
│   ├── document_processor.py   # Document processing and chunking
│   ├── embeddings.py           # Vector embedding functionality
│   ├── rag_pipeline.py         # Main RAG implementation
│   ├── utils.py                # Utility functions
│   └── config.py               # Configuration settings
│
├── app/
│   ├── cli.py                  # Command-line interface
│   └── api.py                  # REST API interface
│
├── notebooks/                  # Jupyter notebooks for exploration
│   └── exploration.ipynb       # Document analysis and testing
│
├── vector_db/                  # Vector database storage (gitignored)
├── main.py                     # Main entry point
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/marbet-assistant.git
   cd marbet-assistant
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

4. Install Tesseract OCR for image text extraction:
   - For Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - For macOS: `brew install tesseract`
   - For Linux: `sudo apt install tesseract-ocr`

5. Place PDF documents in the `data/` directory.

## Usage

### Command Line Interface

```bash
# Run with default settings
python main.py --mode cli

# Rebuild the vector database (first run or when documents change)
python main.py --mode cli --rebuild

# Show source documents for each response
python main.py --mode cli --show-sources

# Specify custom directories
python main.py --mode cli --data-dir ./custom_docs --vector-dir ./custom_vectors
```

### API Server

```bash
# Run API server with default settings
python main.py --mode api

# Specify host and port
python main.py --mode api --host 127.0.0.1 --port 8080

# Rebuild the vector database
python main.py --mode api --rebuild
```

### API Endpoints

- **GET /** - API information
- **POST /api/query** - Process a query
  ```json
  {
    "query": "What activities are available in Halifax?"
  }
  ```
- **GET /api/history** - Get chat history
- **POST /api/clear-history** - Clear chat history

## Configuration Options

Key configuration settings can be adjusted in `src/config.py`:

- **OLLAMA_SERVER**: URL of the Ollama server
- **CHAT_MODEL**: Language model for chat responses
- **EMBED_MODEL**: Model for text embeddings
- **CHUNK_SIZE**: Size of document chunks
- **RETRIEVER_K**: Number of chunks to retrieve per query
- **TEMPERATURE**: Temperature for generation (0.0-1.0)

## Document Categories

The system categorizes documents for better retrieval:

- **activities**: Activity schedules and excursions
- **ship_services**: Onboard amenities and services
- **travel_requirements**: Packing lists and required documents
- **visa_requirements**: ESTA (USA) and eTA (Canada) information
- **technology**: WiFi and device instructions
- **wellness**: Spa services and wellness options

## Development

### Adding New Documents

1. Place new PDF documents in the `data/` directory
2. Run with the `--rebuild` flag to reprocess documents:
   ```bash
   python main.py --mode cli --rebuild
   ```

### Running Tests

```bash
pytest
```

## License

[Insert appropriate license information]

## Contributors

[List of contributors]

## Acknowledgments

This project was developed as part of the Marbet challenge for BUas Advanced Data Science & AI program.
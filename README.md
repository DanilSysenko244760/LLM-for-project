# MARBET Event Assistant

## Overview

MARBET Event Assistant is an AI-powered conversational agent designed to provide detailed information about MARBET's Sales Trip 2024 aboard the Scenic Eclipse I cruise ship. This intelligent assistant uses retrieval-augmented generation (RAG) to answer questions accurately based on MARBET documentation, providing participants with a seamless way to get information about their luxury travel experience.

## Features

- **Semantic Text Processing**: Uses advanced semantic chunking for better understanding of documents and more relevant answers
- **Interactive Web Interface**: User-friendly chat interface with Gradio featuring real-time streaming responses
- **CLI Option**: Command-line interface for lightweight usage scenarios
- **High-Performance RAG**: Optimized vector retrieval with FAISS for fast, relevant responses
- **Source Visualization**: Visual representation of document relevance with automatic citation tracking
- **Document Management**: Easy upload and integration of new documents through the interface
- **Session Persistence**: Maintains conversation context for natural multi-turn interactions
- **Export Functionality**: Export conversations to markdown files
- **Performance Optimizations**: Caching, threading, and efficient memory management

## Installation

### Prerequisites

- Python 3.8+
- Ollama server running (default: http://194.171.191.226:3061)

### Setup

1. Clone the repository:
```bash
git clone [repository_url]
cd marbet-event-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a directory for your MARBET documents:
```bash
mkdir -p docs
```

4. Place your PDF and TXT documents in the docs folder.

## Usage

### Web Interface (Default)

To start the assistant with the web interface:

```python
from main import main

# Run the application with the GUI
if __name__ == "__main__":
    main()
```

Or run the script directly:

```bash
python main.py
```

### Command Line Interface

For a simpler command-line experience:

```python
from main import run_cli

# Run the CLI version
if __name__ == "__main__":
    run_cli()
```

Or via command line arguments:

```bash
python main.py --mode cli
```

### Configuration Options

The `MarbetEventAssistant` class and main script accept several configuration parameters:

| Parameter | Description | Default                            |
|-----------|-------------|------------------------------------|
| `docs_folder` | Directory containing document files | "Should be replaced by local path" |
| `ollama_server` | URL of your Ollama server | "http://194.171.191.226:3061"      |
| `chat_model` | LLM model to use for chat | "llama3.3:70b-instruct-q5_K_M"     |
| `embed_model` | Model to use for embeddings | "mxbai-embed-large:latest"         |
| `cache_dir` | Directory for caching vectors and sessions | "./cache"                          |
| `use_cache` | Whether to use cached vectorstore | False                              |

Example with custom configuration:

```bash
python main.py --docs ./my_documents --server http://localhost:11434 --model mistral:7b --cache ./my_cache --use-cache
```

### Integration in Custom Applications

To use the assistant in your own applications:

```python
from main import MarbetEventAssistant

# Create the assistant with custom settings
assistant = MarbetEventAssistant(
    docs_folder="path/to/your/docs",
    ollama_server="http://your_ollama_server:port",
    chat_model="your_preferred_model",
    embed_model="your_embedding_model",
    cache_dir="./your_cache_dir",
    use_cache=True
)

# Ask a question
result = assistant.ask("What is the itinerary for October 5th?")
print(result["answer"])

# Access source documents if needed
for doc in result["source_documents"]:
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content: {doc.page_content[:100]}...")
```

## Key Components

### SemanticTextSplitter

The application uses a custom text splitter that preserves semantic structure:

- Respects paragraph and sentence boundaries
- Hierarchical splitting (paragraphs first, then sentences if needed)
- Adaptive chunking for varying content lengths
- Fallback strategies for extreme cases

### Efficient Document Processing

- Parallel document loading with ThreadPoolExecutor
- Optimized caching with atomic writes
- Support for PDF and TXT documents

### Streaming Response Handling

- Real-time token streaming for responsive UI
- Buffered updates for performance
- Thread-safe implementation

### Vector Storage and Retrieval

- FAISS for high-performance vector similarity search
- Configurable retrieval parameters
- Caching to avoid reprocessing documents

## Troubleshooting

### Common Issues

1. **Connection Error to Ollama Server**
   - Ensure the Ollama server is running and accessible
   - Check network connectivity and firewall settings
   - Verify correct URL and port in configuration

2. **Document Loading Failures**
   - Check file permissions and encoding
   - Ensure PDF files are not corrupted or password-protected
   - Check the logs for specific file loading issues

3. **Memory Issues**
   - Adjust chunk size and retrieval parameters
   - Use a smaller embedding model
   - Enable caching to reduce memory pressure

4. **Poor Answer Quality**
   - Check that relevant information exists in the document corpus
   - Adjust chunking parameters for better context preservation
   - Try different retrieval parameters (k value)

## Contributing

Contributions to improve the MARBET Event Assistant are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Submit a pull request

---

Developed by [Breda University of Applied Sciences / Oleksii Krasnoshtanov / Danil Sysenko]
# MARBET Event Assistant

## Overview

MARBET Event Assistant is an AI-powered conversational agent specifically designed to provide detailed information about MARBET's Sales Trip 2024 aboard the Scenic Eclipse I cruise ship. This intelligent assistant uses retrieval-augmented generation (RAG) to answer questions accurately based on MARBET documentation, providing participants with a seamless way to get information about their luxury travel experience.

## Features

- **Document-Based Responses**: Answers questions using only information found in MARBET documents
- **Semantic Text Processing**: Uses advanced semantic chunking for better understanding of documents
- **Interactive Web Interface**: User-friendly chat interface with Gradio
- **Source Visualization**: Visual representation of source document relevance
- **Document Management**: Easy upload and integration of new documents
- **Chat History**: Maintains conversation context for natural interactions
- **Export Functionality**: Export conversations to markdown files
- **Stream Responses**: Real-time streaming of AI responses
- **Multi-language Support**: Designed to handle queries in multiple languages (primary: English)

## Installation

### Prerequisites

- Python 3.8+
- Ollama server running (default: http://194.171.191.226:3061)
- Required Python packages (see requirements.txt)

### Setup

1. Clone the repository:
```bash
git clone [repository_url]
cd [repository_name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a directory for your MARBET documents:
```bash
mkdir -p marbet_chatbot/docs
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

Alternatively, modify the `use_gui` flag at the bottom of the script:

```python
if __name__ == "__main__":
    use_gui = False  # Set to False to use CLI instead of GUI
    
    if use_gui:
        main()
    else:
        run_cli()
```

### Command Line Arguments

The assistant accepts several command-line arguments:

```
--mode [gui|cli]     Run in GUI mode (with Gradio interface) or CLI mode
--docs               Path to the documents folder
--server             Ollama server URL
--model              LLM model to use
--embeddings         Embedding model to use
--cache              Cache directory
--use-cache          Use cached vectorstore if available
```

Example:
```bash
python main.py --mode cli --model llama3.3:70b-instruct-q5_K_M --use-cache
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

## Configuration Options

The `MarbetEventAssistant` class accepts several configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `docs_folder` | Directory containing document files | "C:/Users/Alex/Documents/GitHub/LLM-for-project/marbet_chatbot/docs" |
| `ollama_server` | URL of your Ollama server | "http://194.171.191.226:3061" |
| `chat_model` | LLM model to use for chat | "llama3.3:70b-instruct-q5_K_M" |
| `embed_model` | Model to use for embeddings | "mxbai-embed-large:latest" |
| `cache_dir` | Directory for caching vectors and sessions | "./cache" |
| `use_cache` | Whether to use cached vectorstore | False |

## Performance Optimizations

The MARBET Event Assistant includes several optimizations for improved performance:

- **Vectorstore Caching**: Avoids reprocessing documents by caching the vectorstore
- **Semantic Chunking**: Improves retrieval quality through meaningful document splitting
- **Parameterized Retrieval**: The k=12 parameter retrieves sufficient context without overwhelming the model
- **Streaming**: Reduces perceived latency through progressive response rendering
- **ThreadPoolExecutor**: Loads files in parallel for faster document processing
- **Atomic File Operations**: Prevents data corruption during cache writing
- **Efficient String Operations**: Uses optimized string handling techniques
- **Memory Management**: Properly releases resources after visualization creation

## Customization

### System Prompt

You can modify the `SYSTEM_PROMPT` class variable in the `MarbetEventAssistant` class to change the assistant's behavior, tone, and specific instructions. The default prompt is tailored for the MARBET Sales Trip 2024.

### Chunking Strategy

The semantic chunking parameters can be adjusted:
- `chunk_size`: The target size for text chunks (default: 700)
- `chunk_overlap`: The amount of overlap between chunks (default: 150)
- `paragraph_separator`: Define how paragraphs are identified (default: "\n\n")

### Retrieval Parameters

Modify the search parameters in the `_setup_components` method:
- `search_kwargs={"k": 12, "fetch_k": 20}`: Number of chunks to retrieve and candidates to consider

### LLM Parameters

Customize the LLM behavior:
- `temperature`: Controls randomness (higher = more creative, lower = more deterministic)
- `system`: The system prompt that guides the LLM's behavior

## Troubleshooting

### Common Issues

1. **Connection Error to Ollama Server**
   - Ensure the Ollama server is running
   - Check the URL and port are correct
   - Verify network connectivity to the server

2. **Document Loading Failures**
   - Check file permissions
   - Ensure PDF files are not corrupted or password-protected
   - Try converting problematic files to text format

3. **Memory Issues**
   - Reduce chunk size or number of chunks retrieved
   - Use a smaller embedding model
   - Process fewer documents at once

4. **Slow Response Times**
   - Enable caching (`use_cache=True`)
   - Use a faster LLM model
   - Reduce the number of chunks retrieved

5. **Poor Answer Quality**
   - Adjust chunking parameters for better context preservation
   - Increase the number of retrieved chunks
   - Check document quality and formatting
   - Refine the system prompt for better guidance

## Logging

The assistant uses Python's logging module for diagnostics. To adjust the logging level:

```python
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
```

To save logs to a file, modify the logging configuration:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='marbet_assistant.log',
    filemode='a'
)
```

## Contributing

Contributions to improve the MARBET Event Assistant are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Submit a pull request

## License

[Specify your license information here]

---

Developed by [Breda University of Applied Sciences / Oleksii Krasnoshtanov / Danil Sysenko]
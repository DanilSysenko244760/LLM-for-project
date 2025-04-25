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

## Installation

### Prerequisites

- Python 3.8+
- Ollama server running (default: http://194.171.191.226:3061)
- Required Python packages:
  - langchain and langchain_ollama
  - gradio
  - numpy
  - matplotlib
  - PIL
  - nltk
  - faiss-cpu (or faiss-gpu)
  - requests
  - pickle

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
from pr2 import main

# Run the application with the GUI
if __name__ == "__main__":
    main()
```

Or run the script directly:

```bash
python pr2.py
```

### Command Line Interface

For a simpler command-line experience:

```python
from pr2 import run_cli

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

### Integration in Custom Applications

To use the assistant in your own applications:

```python
from pr2 import MarbetEventAssistant

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

## Technical Architecture

### Key Components

1. **SemanticTextSplitter**
   - A custom text splitter that respects semantic boundaries
   - Splits text into chunks based on paragraphs and sentences
   - Preserves context better than simple character-based splitting
   - Handles long sentences by further splitting when necessary
   - Falls back to token-based splitting for extremely long content

2. **StreamHandler**
   - Handles streaming of LLM tokens for real-time response display
   - Captures new tokens and updates the UI accordingly

3. **MarbetEventAssistant**
   - Core class that orchestrates document processing and Q&A
   - Manages document ingestion and vectorization
   - Handles the conversational retrieval chain
   - Maintains chat history and session persistence
   - Provides utilities for visualizing source relevance

4. **Gradio UI Components**
   - Chat interface with message history
   - Document upload capability
   - Source relevance visualization
   - System information display
   - Export and clear functionality

### Document Processing Pipeline

1. Documents are loaded from the specified folder (PDFs and TXT files)
2. Text is extracted and split using semantic boundaries
3. Embeddings are generated for each chunk using the specified embedding model
4. Chunks are stored in a FAISS vector database for efficient retrieval
5. User queries retrieve relevant chunks from the vector database
6. LLM generates responses based on retrieved context and conversation history

## Customization

### System Prompt

You can modify the `SYSTEM_PROMPT` class variable in the `MarbetEventAssistant` class to change the assistant's behavior, tone, and specific instructions. The default prompt is tailored for the MARBET Sales Trip 2024, but you can customize it for other events or purposes.

### Chunking Strategy

The semantic chunking parameters can be adjusted:
- `chunk_size`: The target size for text chunks (default: 700)
- `chunk_overlap`: The amount of overlap between chunks (default: 150)
- `paragraph_separator`: Define how paragraphs are identified (default: "\n\n")

### Retrieval Parameters

Modify the search parameters in the `_setup_components` method:
- `search_kwargs={"k": 12}`: Number of chunks to retrieve (adjust for more or less context)

### LLM Parameters

Customize the LLM behavior:
- `temperature`: Controls randomness (higher = more creative, lower = more deterministic)
- `system`: The system prompt that guides the LLM's behavior
- Other parameters can be passed to the OllamaLLM constructor

### UI Customization

The Gradio interface can be customized by modifying the `create_gradio_interface` function:
- Change theme, layout, and component properties
- Add or remove interface elements
- Modify chat avatar images
- Adjust visualization settings

## Advanced Usage

### Adding New Documents Programmatically

```python
assistant = MarbetEventAssistant()
result = assistant.add_document("/path/to/new/document.pdf")
print(result)  # Should show success message
```

### Exporting Chat History

```python
assistant = MarbetEventAssistant()
# ... ask some questions ...
export_path = assistant.export_chat()
print(f"Chat exported to: {export_path}")
```

### Visualizing Source Documents

```python
assistant = MarbetEventAssistant()
result = assistant.ask("What is the itinerary?")
viz_path = assistant.visualize_sources(result["source_documents"])
print(f"Visualization saved to: {viz_path}")
```

## Troubleshooting

### Common Issues

1. **Connection Error to Ollama Server**
   - Ensure the Ollama server is running
   - Check the URL and port are correct
   - Verify network connectivity to the server
   - Try running a simple query directly to the Ollama server to verify it's responsive

2. **Document Loading Failures**
   - Check file permissions
   - Ensure PDF files are not corrupted or password-protected
   - Try converting problematic files to text format
   - Check the error logs for specific file loading issues

3. **Memory Issues**
   - Reduce chunk size or number of chunks retrieved
   - Use a smaller embedding model
   - Process fewer documents at once
   - Increase chunk size to reduce the total number of chunks

4. **Slow Response Times**
   - Enable caching (`use_cache=True`)
   - Use a faster LLM model
   - Reduce the number of chunks retrieved
   - Check server resources (CPU, RAM, GPU)

5. **Poor Answer Quality**
   - Adjust chunking parameters for better context preservation
   - Increase the number of retrieved chunks
   - Check document quality and formatting
   - Refine the system prompt for better guidance

## Logging

The assistant uses Python's logging module for diagnostics. Logs are output with timestamp, level, and message information. You can adjust the logging level in the code by modifying:

```python
logging.basicConfig(level=logging.INFO)  # Change to logging.DEBUG for more detail
```

Log files are not stored by default, but you can modify the logging configuration to save logs:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='marbet_assistant.log',
    filemode='a'
)
```

## Session Management

Chat sessions are automatically saved in the cache directory and can be exported to markdown files. Each session has a unique ID based on the start time.

## Code Structure

- **SemanticTextSplitter Class**: Custom implementation of LangChain's TextSplitter
- **StreamHandler Class**: Custom callback handler for streaming responses
- **MarbetEventAssistant Class**: Main assistant implementation with methods for document processing, querying, and utilities
- **create_gradio_interface Function**: Creates the Gradio UI for the assistant
- **main Function**: Entry point for the web interface application
- **run_cli Function**: Entry point for the command-line interface

## Contributing

Contributions to improve the MARBET Event Assistant are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Submit a pull request

---

Developed by [Breda University of applied sciences / Oleksii Krasnoshtanov / Danil Sysenko]

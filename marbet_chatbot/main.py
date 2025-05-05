import os
import time
import argparse
import functools
import requests
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import pickle
import logging
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor
import threading

# Updated LangChain imports
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import TextSplitter

# Configure logging with a more efficient format
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Import and configure NLTK with try-except - using a singleton pattern for nltk data
_nltk_initialized = False


def initialize_nltk():
    global _nltk_initialized
    if _nltk_initialized:
        return True

    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        # Download required NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # Download punkt instead of punkt_tab
            nltk.download('punkt', quiet=True)

            # Verify it was downloaded successfully
            try:
                nltk.data.find('tokenizers/punkt')
                logger.info("NLTK punkt downloaded successfully")
            except LookupError:
                logger.warning("Failed to download NLTK punkt data")
                return False

        _nltk_initialized = True
        return True
    except ImportError:
        logger.warning("NLTK not available, using simple sentence tokenizer")
        return False


# Define a simple fallback sentence tokenizer
def simple_sentence_tokenizer(text):
    # Optimized regex-based tokenizer with precompiled pattern
    pattern = re.compile(r'(?<=[.!?])\s+')
    return pattern.split(text)


# Get appropriate sentence tokenizer
def get_sent_tokenize():
    if initialize_nltk():
        import nltk
        return nltk.tokenize.sent_tokenize
    else:
        return simple_sentence_tokenizer


class SemanticTextSplitter(TextSplitter):
    """A text splitter that uses semantic boundaries like paragraphs, sections, and sentences."""

    def __init__(
            self,
            chunk_size: int = 700,
            chunk_overlap: int = 150,
            paragraph_separator: str = "\n\n",
            sentence_separator: Optional[str] = None,
            keep_separator: bool = True,
            length_function: callable = len,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                         length_function=length_function, keep_separator=keep_separator)
        self.paragraph_separator = paragraph_separator
        self.sentence_separator = sentence_separator
        self.sent_tokenize = get_sent_tokenize()
        # Precompile the pattern for empty paragraphs
        self._empty_pattern = re.compile(r"^\s*$")

    def split_text(self, text: str) -> List[str]:
        # First split by paragraphs
        if self.paragraph_separator:
            paragraphs = text.split(self.paragraph_separator)
        else:
            paragraphs = [text]

        # Initialize output list
        chunks = []
        current_chunk = []
        current_chunk_len = 0

        for paragraph in paragraphs:
            # Skip empty paragraphs more efficiently
            if self._empty_pattern.match(paragraph):
                continue

            paragraph_len = self._length_function(paragraph)

            # If the paragraph fits within chunk_size, keep it as a unit
            if paragraph_len <= self._chunk_size:
                if current_chunk_len + paragraph_len <= self._chunk_size:
                    # Add to current chunk
                    current_chunk.append(paragraph)
                    current_chunk_len += paragraph_len
                else:
                    # Start a new chunk
                    if current_chunk:
                        chunks.append(self.paragraph_separator.join(current_chunk)
                                      if self._keep_separator else "".join(current_chunk))
                    current_chunk = [paragraph]
                    current_chunk_len = paragraph_len
            else:
                # Paragraph is too large, split into sentences
                try:
                    # Try using NLTK's sentence tokenizer first
                    if not self.sentence_separator:
                        sentences = self.sent_tokenize(paragraph)
                    else:
                        sentences = paragraph.split(self.sentence_separator)
                except Exception as e:
                    # If tokenization fails, use our simple tokenizer
                    logger.warning(f"Sentence tokenization failed: {e}, using simple tokenizer")
                    sentences = simple_sentence_tokenizer(paragraph)

                # Process each sentence
                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    sentence_len = self._length_function(sentence)

                    # Handle case where a single sentence is too large
                    if sentence_len > self._chunk_size:
                        # If we have accumulated text, add it to chunks
                        if current_chunk:
                            chunks.append(self.paragraph_separator.join(current_chunk)
                                          if self._keep_separator else "".join(current_chunk))
                            current_chunk = []
                            current_chunk_len = 0

                        # Split the long sentence using simple splitting
                        # Creating TokenTextSplitter is expensive, so only do it when needed
                        token_splitter = TokenTextSplitter(
                            chunk_size=self._chunk_size,
                            chunk_overlap=self._chunk_overlap
                        )
                        sub_chunks = token_splitter.split_text(sentence)
                        chunks.extend(sub_chunks)
                    else:
                        # Normal case - add sentence to current chunk if it fits
                        if current_chunk_len + sentence_len <= self._chunk_size:
                            current_chunk.append(sentence)
                            current_chunk_len += sentence_len
                        else:
                            # Finalize current chunk and start a new one
                            if current_chunk:
                                chunks.append(self.paragraph_separator.join(current_chunk)
                                              if self._keep_separator else "".join(current_chunk))
                            current_chunk = [sentence]
                            current_chunk_len = sentence_len

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(self.paragraph_separator.join(current_chunk)
                          if self._keep_separator else "".join(current_chunk))

        return chunks


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.update_lock = threading.Lock()
        self.buffer = []
        self.last_update = time.time()

    def on_llm_new_token(self, token: str, **kwargs):
        with self.update_lock:
            self.text += token
            self.buffer.append(token)

            # Update UI every 100ms or when buffer gets large
            current_time = time.time()
            if current_time - self.last_update > 0.1 or len(self.buffer) > 10:
                self.container.update(value=self.text)
                self.buffer = []
                self.last_update = current_time


class MarbetEventAssistant:
    SYSTEM_PROMPT = """You are MARBET's AI-powered Event Assistant for the Sales Trip 2024. Your primary purpose is to provide accurate, helpful information to participants about their upcoming luxury travel experience on the Scenic Eclipse I cruise ship, traveling from Canada to the USA in October 2024.
Core Principles

Document-Based Responses Only: Answer questions using ONLY information found in the provided MARBET documents. Never invent or guess information not explicitly stated in these materials. 
Event Context: You assist with a luxury cruise sales trip from October 4-11, 2024, traveling from Halifax, Canada to New York City, with stops in Lunenburg, Portland, Boston, Provincetown, and Martha's Vineyard.
"Election Program" Clarification: The term "Election program" in all documents refers EXCLUSIVELY to optional activities participants can select - it has NOTHING to do with political elections. Always interpret this term as optional excursion choices.
Accuracy with Dates and Locations: When users ask about specific dates (e.g., "Friday, 11.10.2024") or locations (e.g., "New York City"), search for those exact terms in the documents and provide the relevant information.
Clear Labeling of Assumptions: If making a logical assumption based on document context, explicitly label it: "Based on the documentation, I would assume that..."
Handling Missing Information:

Acknowledge understanding: "I understand you're asking about [topic]."
Be transparent: "The MARBET documents don't provide specific information about [topic]."
Suggest next steps: "For the most accurate information, please contact the crew team."


Communication Style:

Friendly and welcoming
Concise and clear
Natural conversational tone
Professional but not overly formal
Enthusiastic about the luxury travel experience



Key Information Areas

Itinerary: Assist with dates, locations, and scheduled activities for each port
Optional Activities: Provide details on "Election program" options at each location
Ship Information: Answer questions about the Scenic Eclipse I facilities and services
Travel Documentation: Guide on ESTA/eTA requirements for USA/Canada
Packing & Preparation: Advise on what to bring based on the packing list
Technical Assistance: Help with WiFi connection and onboard technology
Safety & Procedures: Explain ship safety protocols and emergency procedures

Response Structure

Acknowledge the question with a friendly greeting
Provide a direct, accurate answer based exclusively on MARBET documents
Add relevant context or additional helpful information if available in documents
Clarify any assumptions you're making, clearly labeled as such
End with an offer to help with further questions

Remember that your goal is to make participants feel informed, confident, and excited about their upcoming luxury travel experience, while sticking strictly to the factual information provided in the MARBET documents."""

    def __init__(self,
                 docs_folder: str = "C:/Users/Alex/Documents/GitHub/LLM-for-project/marbet_chatbot/docs",
                 ollama_server: str = "http://194.171.191.226:3061",
                 chat_model: str = 'llama3.3:70b-instruct-q5_K_M',
                 embed_model: str = "mxbai-embed-large:latest",
                 cache_dir: str = "./cache",
                 use_cache: bool = False):
        self.docs_folder = docs_folder
        self.ollama_server = ollama_server
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.cache_dir = cache_dir
        self.vectorstore = None
        self.qa_chain = None
        self.chat_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_cache = use_cache
        self._setup_lock = threading.Lock()

        # Create directories only once during initialization
        os.makedirs(docs_folder, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "sessions"), exist_ok=True)

        self._setup_components()

    @functools.lru_cache(maxsize=32)
    def _get_cache_path(self):
        """Cache the cache path to avoid repeated string operations"""
        return os.path.join(self.cache_dir, "vectorstore.pkl")

    def _setup_components(self):
        with self._setup_lock:
            cache_path = self._get_cache_path()
            if self.use_cache and os.path.exists(cache_path):
                try:
                    logger.info("Loading vectorstore from cache...")
                    with open(cache_path, 'rb') as f:
                        self.vectorstore = pickle.load(f)
                    logger.info("Vectorstore loaded successfully!")
                except Exception as e:
                    logger.error(f"Error loading cache: {e}")
                    self._process_documents()
            else:
                self._process_documents()

            # Only create retriever and LLM if they don't exist already
            if not hasattr(self, 'retriever') or self.retriever is None:
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 12, "fetch_k": 20}  # Fetch more candidates for better filtering
                )

            if not hasattr(self, 'llm') or self.llm is None:
                self.llm = OllamaLLM(
                    base_url=self.ollama_server,
                    model=self.chat_model,
                    temperature=0.8,
                    system=self.SYSTEM_PROMPT
                )

            if not hasattr(self, 'qa_chain') or self.qa_chain is None:
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    return_source_documents=True
                )

    def _process_documents(self):
        logger.info("Processing documents...")

        # Use ThreadPoolExecutor to load files in parallel
        all_docs = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both loaders to run in parallel
            pdf_future = executor.submit(self._load_pdfs)
            txt_future = executor.submit(self._load_txts)

            # Get results from both futures
            pdf_docs = pdf_future.result()
            txt_docs = txt_future.result()

            all_docs.extend(pdf_docs)
            all_docs.extend(txt_docs)

        # Split documents using semantic chunking with optimized settings
        splitter = SemanticTextSplitter(chunk_size=700, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)
        logger.info(f"Split documents into {len(chunks)} chunks using semantic chunking")

        # Create embeddings and vectorstore
        embeddings = OllamaEmbeddings(
            base_url=self.ollama_server,
            model=self.embed_model
        )
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("Created vectorstore from document chunks")

        # Skip caching if the environment variable is set
        if os.environ.get("SKIP_CACHE", "0") == "1":
            logger.info("Skipping vectorstore caching due to environment setting")
            return

        # Cache the vectorstore
        self._save_vectorstore()

    def _load_pdfs(self):
        """Load PDF files with error handling"""
        try:
            pdf_loader = PyPDFDirectoryLoader(self.docs_folder, extract_images=False)
            docs = pdf_loader.load()
            logger.info(f"Loaded {len(docs)} PDF documents from {self.docs_folder}")
            return docs
        except Exception as e:
            logger.error(f"Error loading PDF files: {e}")
            return []

    def _load_txts(self):
        """Load TXT files with error handling"""
        try:
            txt_loader = DirectoryLoader(
                self.docs_folder,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            docs = txt_loader.load()
            logger.info(f"Loaded {len(docs)} TXT documents from {self.docs_folder}")
            return docs
        except Exception as e:
            logger.error(f"Error loading text files: {e}")
            return []

    def _save_vectorstore(self):
        """Save vectorstore to cache with error handling"""
        cache_path = self._get_cache_path()
        temp_path = f"{cache_path}.tmp"
        try:
            # Write to temporary file first
            with open(temp_path, 'wb') as f:
                pickle.dump(self.vectorstore, f, protocol=pickle.HIGHEST_PROTOCOL)

            # If successful, rename to final path (atomic operation)
            os.replace(temp_path, cache_path)
            logger.info("Vectorstore cached successfully!")
        except Exception as e:
            logger.warning(f"Error caching vectorstore: {e}")
            logger.info("Continuing without caching. Performance may be slower on restart.")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def add_document(self, file_path):
        """Add a document to the vectorstore with better error handling and optimization"""
        try:
            # Validate file exists
            if not os.path.isfile(file_path):
                return f"File not found: {os.path.basename(file_path)}"

            # Check file type
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path, extract_images=False)
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                return f"Unsupported file format: {os.path.basename(file_path)}"

            # Load document
            docs = loader.load()
            if not docs:
                return f"No content found in {os.path.basename(file_path)}"

            # Use semantic chunking
            splitter = SemanticTextSplitter(chunk_size=700, chunk_overlap=150)
            chunks = splitter.split_documents(docs)

            # Get embeddings and add to vectorstore
            embeddings = OllamaEmbeddings(base_url=self.ollama_server, model=self.embed_model)
            self.vectorstore.add_documents(chunks)

            # Update cache
            self._save_vectorstore()

            # Update retriever to use new documents
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 12})

            return f"Document {os.path.basename(file_path)} added successfully with {len(chunks)} chunks."
        except Exception as e:
            logger.error(f"Error adding document: {e}", exc_info=True)
            return f"Error adding document: {str(e)}"

    def ask(self, query: str, stream_container=None) -> Dict[str, Any]:
        """Process a query with optimized streaming and error handling"""
        if not query or not query.strip():
            return {"answer": "I didn't receive a question. Could you please ask me something?",
                    "source_documents": []}

        try:
            # Create callback handler for streaming if container provided
            callbacks = []
            if stream_container:
                stream_handler = StreamHandler(stream_container)
                callbacks.append(stream_handler)

            # Process the query
            result = self.qa_chain.invoke(
                {
                    "question": query,
                    "chat_history": self.chat_history
                },
                callbacks=callbacks if callbacks else None
            )

            # Update chat history - limit to 10 entries for memory efficiency
            self.chat_history.append((query, result["answer"]))
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

            # Save session asynchronously
            threading.Thread(target=self._save_session).start()

            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            error_msg = f"I encountered an error processing your question. Please try again or rephrase your question."
            return {"answer": error_msg, "source_documents": []}

    def _save_session(self):
        """Save session data with error handling and atomic writes"""
        session_path = os.path.join(self.cache_dir, "sessions", f"{self.session_id}.pkl")
        temp_path = f"{session_path}.tmp"
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(self.chat_history, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Atomic replace
            os.replace(temp_path, session_path)
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            # Clean up temp file if necessary
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    @functools.lru_cache(maxsize=8)
    def _get_color_map(self, count):
        """Cache the color map generation to avoid recomputing for similar source counts"""
        return plt.cm.Blues(np.linspace(0.6, 1.0, count))

    def visualize_sources(self, source_documents):
        """Generate visualization of source documents with caching and optimization"""
        if not source_documents:
            return None

        # Limit to top 10 sources for better visualization
        if len(source_documents) > 10:
            source_documents = source_documents[:10]

        # Extract source names and clean them up
        source_names = []
        for doc in source_documents:
            source = doc.metadata.get('source', 'Unknown')
            # Get just the filename without the full path
            source_name = os.path.basename(source)
            # Truncate very long filenames
            if len(source_name) > 30:
                source_name = source_name[:27] + "..."
            source_names.append(source_name)

        # Use exponential decay for confidence scores (more pronounced differences)
        confidence = np.linspace(0.9, 0.5, len(source_names))
        colors = self._get_color_map(len(source_names))

        # Create figure with smaller memory footprint
        plt.figure(figsize=(10, 6), dpi=80)
        bars = plt.barh(source_names, confidence, color=colors)
        plt.xlabel('Relevance Score')
        plt.title('Source Documents Relevance')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Add text labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{confidence[i]:.2f}', va='center')

        # Save and close to free memory
        plt.tight_layout()
        temp_path = os.path.join(self.cache_dir, f"source_viz_{int(time.time())}.png")
        plt.savefig(temp_path, dpi=80, bbox_inches='tight')
        plt.close('all')  # Explicitly close all figures

        return temp_path

    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        # Create new session ID for fresh start
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return "Conversation history cleared."

    def export_chat(self):
        """Export the chat history to a markdown file with optimization"""
        if not self.chat_history:
            return None

        # Use string builder pattern for more efficient string concatenation
        lines = ["# Marbet Event Assistant - Chat Export\n\n"]
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        for query, answer in self.chat_history:
            lines.append(f"## User\n{query}\n\n## Assistant\n{answer}\n\n---\n\n")

        export_path = os.path.join(self.cache_dir, f"chat_export_{self.session_id}.md")

        # Write efficiently with a single operation
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(''.join(lines))

        return export_path


# Gradio UI component - optimized for efficiency
def create_gradio_interface(assistant):
    """Create an optimized Gradio interface for the assistant"""

    # Preload avatar images to avoid repeated network requests
    user_avatar = "https://cdn-icons-png.flaticon.com/512/5231/5231812.png"
    bot_avatar = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"

    # Use a more efficient theme setting
    theme = gr.themes.Soft()

    with gr.Blocks(title="Marbet Event Assistant", theme=theme) as interface:
        gr.Markdown(
            """
            # 🎪 Marbet Event Assistant

            ## Your AI guide to Marbet events and services

            Ask me questions about event planning, venues, catering, entertainment options, and more!
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                # Optimize chatbot component
                chatbot = gr.Chatbot(
                    height=500,
                    bubble_full_width=False,
                    show_copy_button=True,
                    avatar_images=(user_avatar, bot_avatar)
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about Marbet services...",
                        container=False,
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", scale=1)
                    export_btn = gr.Button("Export Chat", scale=1)

            with gr.Column(scale=2):
                with gr.Accordion("Source Information", open=True):
                    source_image = gr.Image(label="Relevance of Sources", type="filepath")

                with gr.Accordion("Document Management", open=False):
                    upload_btn = gr.File(label="Upload New Document")
                    upload_status = gr.Textbox(label="Upload Status")

                with gr.Accordion("System Information", open=False):
                    # Use simpler JSON structure to reduce memory usage
                    sys_info = gr.JSON({
                        "Model": assistant.chat_model,
                        "Embedding": assistant.embed_model,
                        "Session ID": assistant.session_id
                    })

        # Response object for streaming
        response_obj = gr.Textbox(visible=False)

        # Optimize event handlers
        def user_input(message, history):
            """Handle user input efficiently"""
            if not message or not message.strip():
                return "", history
            return "", history + [[message, None]]

        def bot_response(history):
            """Process bot response with optimized source visualization"""
            if not history:
                return history, None

            query = history[-1][0]
            # Get response from assistant
            result = assistant.ask(query, response_obj)

            # Update the source visualization
            viz_path = None
            if result.get("source_documents"):
                viz_path = assistant.visualize_sources(result.get("source_documents", []))

            # Update history
            history[-1][1] = result["answer"]
            return history, viz_path

        def handle_upload(file):
            """Handle document upload with validation"""
            if file is None:
                return "No file selected."

            # Validate file type
            filename = file.name.lower()
            if not (filename.endswith('.pdf') or filename.endswith('.txt')):
                return "Only PDF and TXT files are supported."

            # Save the file temporarily
            temp_path = os.path.join(assistant.cache_dir, file.name)
            try:
                with open(temp_path, 'wb') as f:
                    f.write(file.read())

                # Add to assistant
                result = assistant.add_document(temp_path)

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            return result

        def export_chat():
            """Export chat history"""
            export_path = assistant.export_chat()
            if not export_path:
                return gr.File.update(value=None, visible=False)
            return gr.File.update(value=export_path, visible=True)

        # Set up UI interactions with optimized event flow
        msg.submit(
            user_input,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            chatbot,
            [chatbot, source_image]
        )

        submit_btn.click(
            user_input,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            chatbot,
            [chatbot, source_image]
        )

        # More efficient clear operation
        clear_btn.click(lambda: ([], None), None, [chatbot, source_image], queue=False)
        clear_btn.click(assistant.clear_history, None, None)

        # Document upload handling
        upload_btn.change(handle_upload, upload_btn, upload_status)

        # Export functionality
        export_file = gr.File(label="Exported Chat", visible=False)
        export_btn.click(export_chat, None, export_file)

    return interface


def main(mode='gui', **kwargs):
    """Main function to run the assistant with better error handling"""
    try:
        # Create the assistant with passed arguments
        assistant = MarbetEventAssistant(**kwargs)

        if mode.lower() == 'cli':
            run_cli(assistant)
        else:
            # Create and launch the Gradio interface with optimized settings
            interface = create_gradio_interface(assistant)
            interface.launch(
                share=True,
                show_error=True,
                server_name="0.0.0.0",  # Bind to all interfaces
                quiet=True  # Reduce console output
            )
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        print(f"Error: {e}")
        print(f"Detailed error information: {e.__class__.__name__}")
        import traceback
        traceback.print_exc()


# Command-line version with optimization
def run_cli(assistant=None):
    """Run the assistant in command line mode with improved UI"""
    if assistant is None:
        assistant = MarbetEventAssistant()

    print("=" * 50)
    print("MARBET EVENT ASSISTANT".center(50))
    print("=" * 50)
    print("Ask me something about the documents. Type 'exit' to quit.")

    while True:
        try:
            query = input("\nYou: ")
            if not query.strip():
                continue

            if query.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye! Thank you for using the Marbet Event Assistant.")
                break

            print("\nAssistant: ", end="", flush=True)

            # Simple streaming in CLI mode
            result = assistant.ask(query)

            # Print the result character by character for a streaming effect
            for char in result["answer"]:
                print(char, end="", flush=True)
                time.sleep(0.005)  # Small delay for readability
            print()

            # Print sources
            if result.get("source_documents"):
                sources = set([os.path.basename(doc.metadata.get('source', 'Unknown'))
                               for doc in result.get("source_documents", [])])
                if sources:
                    print("\nSources: " + ", ".join(list(sources)[:5]))
                    if len(sources) > 5:
                        print(f"and {len(sources) - 5} more...")

        except KeyboardInterrupt:
            print("\n\nOperation interrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\nError occurred: {str(e)}")


if __name__ == "__main__":
    # Parse command line arguments with improved default values
    parser = argparse.ArgumentParser(description="Marbet Event Assistant")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui",
                        help="Run in GUI mode (with Gradio interface) or CLI mode")
    parser.add_argument("--docs",
                        help="Path to the documents folder")
    parser.add_argument("--server",
                        help="Ollama server URL")
    parser.add_argument("--model",
                        help="LLM model to use")
    parser.add_argument("--embeddings",
                        help="Embedding model to use")
    parser.add_argument("--cache",
                        help="Cache directory")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached vectorstore if available")

    args = parser.parse_args()

    # Build kwargs dictionary with only provided args
    kwargs = {}
    if args.docs:
        kwargs['docs_folder'] = args.docs
    if args.server:
        kwargs['ollama_server'] = args.server
    if args.model:
        kwargs['chat_model'] = args.model
    if args.embeddings:
        kwargs['embed_model'] = args.embeddings
    if args.cache:
        kwargs['cache_dir'] = args.cache
    if args.use_cache:
        kwargs['use_cache'] = True

    # Run with the appropriate mode
    main(mode=args.mode, **kwargs)
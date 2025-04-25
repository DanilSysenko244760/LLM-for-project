import os
import time
import requests
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import pickle
import logging
from datetime import datetime

# LangChain imports
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import TextSplitter
from typing import List, Optional, Any
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
            # Skip empty paragraphs
            if not paragraph.strip():
                continue

            # If the paragraph fits within chunk_size, keep it as a unit
            if self.length_function(paragraph) <= self.chunk_size:
                if current_chunk_len + self.length_function(paragraph) <= self.chunk_size:
                    # Add to current chunk
                    current_chunk.append(paragraph)
                    current_chunk_len += self.length_function(paragraph)
                else:
                    # Start a new chunk
                    if current_chunk:
                        chunks.append(self.paragraph_separator.join(current_chunk)
                                      if self.keep_separator else "".join(current_chunk))
                    current_chunk = [paragraph]
                    current_chunk_len = self.length_function(paragraph)
            else:
                # Paragraph is too large, split into sentences
                if not self.sentence_separator:
                    # Use NLTK to split into sentences
                    sentences = sent_tokenize(paragraph)
                else:
                    sentences = paragraph.split(self.sentence_separator)

                # Process each sentence
                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    sentence_len = self.length_function(sentence)

                    # Handle case where a single sentence is too large
                    if sentence_len > self.chunk_size:
                        # If we have accumulated text, add it to chunks
                        if current_chunk:
                            chunks.append(self.paragraph_separator.join(current_chunk)
                                          if self.keep_separator else "".join(current_chunk))
                            current_chunk = []
                            current_chunk_len = 0

                        # Split the long sentence using simple splitting
                        token_splitter = TokenTextSplitter(
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap
                        )
                        sub_chunks = token_splitter.split_text(sentence)
                        chunks.extend(sub_chunks)
                    else:
                        # Normal case - add sentence to current chunk if it fits
                        if current_chunk_len + sentence_len <= self.chunk_size:
                            current_chunk.append(sentence)
                            current_chunk_len += sentence_len
                        else:
                            # Finalize current chunk and start a new one
                            if current_chunk:
                                chunks.append(self.paragraph_separator.join(current_chunk)
                                              if self.keep_separator else "".join(current_chunk))
                            current_chunk = [sentence]
                            current_chunk_len = sentence_len

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(self.paragraph_separator.join(current_chunk)
                          if self.keep_separator else "".join(current_chunk))

        return chunks


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.update(value=self.text)


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

        os.makedirs(docs_folder, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "sessions"), exist_ok=True)

        self._setup_components()

    def _setup_components(self):
        cache_path = os.path.join(self.cache_dir, "vectorstore.pkl")
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

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 12})
        self.llm = OllamaLLM(
            base_url=self.ollama_server,
            model=self.chat_model,
            temperature=0.8,
            system=self.SYSTEM_PROMPT
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

    def _process_documents(self):
        logger.info("Processing documents...")

        all_docs = []

        # Load PDFs
        try:
            pdf_loader = PyPDFDirectoryLoader(self.docs_folder, extract_images=False)
            all_docs.extend(pdf_loader.load())
            logger.info(f"Loaded PDF files from {self.docs_folder}")
        except Exception as e:
            logger.error(f"Error loading PDF files: {e}")

        # Load TXT files
        try:
            txt_loader = DirectoryLoader(
                self.docs_folder,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            all_docs.extend(txt_loader.load())
            logger.info(f"Loaded TXT files from {self.docs_folder}")
        except Exception as e:
            logger.error(f"Error loading text files: {e}")

        # Split documents using semantic chunking
        splitter = SemanticTextSplitter(chunk_size=700, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)
        logger.info(f"Split documents into {len(chunks)} chunks using semantic chunking")

        # Create embeddings and vectorstore
        embeddings = OllamaEmbeddings(base_url=self.ollama_server, model=self.embed_model)
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("Created vectorstore from document chunks")

        # Cache the vectorstore
        cache_path = os.path.join(self.cache_dir, "vectorstore.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.vectorstore, f)
            logger.info("Vectorstore cached successfully!")
        except Exception as e:
            logger.error(f"Error caching vectorstore: {e}")

    def add_document(self, file_path):
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path, extract_images=False)
            elif file_path.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                return f"Unsupported file format: {os.path.basename(file_path)}"

            docs = loader.load()

            # Use semantic chunking instead of character-based splitting
            splitter = SemanticTextSplitter(chunk_size=700, chunk_overlap=150)
            chunks = splitter.split_documents(docs)

            embeddings = OllamaEmbeddings(base_url=self.ollama_server, model=self.embed_model)
            self.vectorstore.add_documents(chunks)

            cache_path = os.path.join(self.cache_dir, "vectorstore.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.vectorstore, f)

            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 12})

            return f"Document {os.path.basename(file_path)} added successfully."
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return f"Error adding document: {str(e)}"

    def ask(self, query: str, stream_container=None) -> Dict[str, Any]:
        try:
            callbacks = []
            if stream_container:
                stream_handler = StreamHandler(stream_container)
                callbacks.append(stream_handler)

            result = self.qa_chain.invoke(
                {
                    "question": query,
                    "chat_history": self.chat_history
                },
                callbacks=callbacks if callbacks else None
            )

            self.chat_history.append((query, result["answer"]))

            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

            self._save_session()

            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"answer": f"I encountered an error: {str(e)}", "source_documents": []}

    def _save_session(self):
        session_path = os.path.join(self.cache_dir, "sessions", f"{self.session_id}.pkl")
        try:
            with open(session_path, 'wb') as f:
                pickle.dump(self.chat_history, f)
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def visualize_sources(self, source_documents):
        if not source_documents:
            return None

        source_names = [os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in source_documents]
        confidence = np.linspace(0.9, 0.5, len(source_names))
        colors = plt.cm.Blues(np.linspace(0.6, 1.0, len(source_names)))

        plt.figure(figsize=(10, 6))
        bars = plt.barh(source_names, confidence, color=colors)
        plt.xlabel('Relevance Score')
        plt.title('Source Documents Relevance')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{confidence[i]:.2f}', va='center')

        plt.tight_layout()
        temp_path = os.path.join(self.cache_dir, f"source_viz_{int(time.time())}.png")
        plt.savefig(temp_path)
        plt.close()

        return temp_path

    def clear_history(self):
        self.chat_history = []
        return "Conversation history cleared."

    def export_chat(self):
        """Export the chat history to a markdown file"""
        export_text = "# Marbet Event Assistant - Chat Export\n\n"
        export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

        for query, answer in self.chat_history:
            export_text += f"## User\n{query}\n\n## Assistant\n{answer}\n\n---\n\n"

        export_path = os.path.join(self.cache_dir, f"chat_export_{self.session_id}.md")

        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(export_text)

        return export_path


# Gradio UI component
def create_gradio_interface(assistant):
    """Create a Gradio interface for the assistant"""
    with gr.Blocks(title="Marbet Event Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🎪 Marbet Event Assistant

            ## Your AI guide to Marbet events and services

            Ask me questions about event planning, venues, catering, entertainment options, and more!
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    bubble_full_width=False,
                    show_copy_button=True,
                    avatar_images=(
                        "https://cdn-icons-png.flaticon.com/512/5231/5231812.png",
                        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
                    )
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
                    sys_info = gr.JSON({
                        "Model": assistant.chat_model,
                        "Embedding Model": assistant.embed_model,
                        "Server": assistant.ollama_server,
                        "Documents Location": assistant.docs_folder,
                        "Session ID": assistant.session_id
                    })

        # Response for streaming
        response_obj = gr.Textbox(visible=False)

        # Define event handlers
        def user_input(message, history):
            return "", history + [[message, None]]

        def bot_response(history):
            query = history[-1][0]
            result = assistant.ask(query, response_obj)

            # Update the source visualization
            viz_path = assistant.visualize_sources(result.get("source_documents", []))

            history[-1][1] = result["answer"]
            return history, viz_path

        def handle_upload(file):
            if file is None:
                return "No file selected."

            # Save the file temporarily
            temp_path = os.path.join(assistant.cache_dir, file.name)

            with open(temp_path, 'wb') as f:
                f.write(file.read())

            # Add to assistant
            result = assistant.add_document(temp_path)

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return result

        def export_chat():
            export_path = assistant.export_chat()
            return gr.File.update(value=export_path, visible=True)

        # Set up UI interactions
        msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, [chatbot, source_image]
        )

        submit_btn.click(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, [chatbot, source_image]
        )

        clear_btn.click(lambda: ([], None), None, [chatbot, source_image], queue=False)
        clear_btn.click(assistant.clear_history, None, None)

        upload_btn.change(handle_upload, upload_btn, upload_status)

        export_file = gr.File(label="Exported Chat", visible=False)
        export_btn.click(export_chat, None, export_file)

    return interface


def main():
    """Main function to run the assistant"""
    try:
        # Create the assistant
        assistant = MarbetEventAssistant()

        # Create and launch the Gradio interface
        interface = create_gradio_interface(assistant)
        interface.launch(share=True)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        print(f"Error: {e}")
        print(f"Detailed error information: {e.__class__.__name__}")
        import traceback
        traceback.print_exc()


# Also include a command-line version if needed
def run_cli():
    """Run the assistant in command line mode"""
    assistant = MarbetEventAssistant()
    print("=" * 50)
    print("MARBET EVENT ASSISTANT".center(50))
    print("=" * 50)
    print("Ask me something about the documents. Type 'exit' to quit.")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            result = assistant.ask(query)
            print("Assistant:", result["answer"])
        except Exception as e:
            print("Error occurred:", str(e))


if __name__ == "__main__":
    # Choose which version to run
    use_gui = True  # Set to False to use CLI instead of GUI

    if use_gui:
        main()
    else:
        run_cli()
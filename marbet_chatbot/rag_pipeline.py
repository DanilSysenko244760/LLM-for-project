from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG Pipeline for the Marbet chatbot"""

    def __init__(self, ollama_url: str = "http://194.171.191.226:3061",
                 model_name: str = "llama3.1:8b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None

    def setup_vectorstore(self, document_chunks: List, persist_dir: str = "./chroma_db"):
        """Set up the vector store for document retrieval"""
        logger.info(f"Setting up vector store with {len(document_chunks)} chunks")

        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            base_url=self.ollama_url,
            model=self.model_name
        )

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=document_chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )

        # Persist to disk
        self.vectorstore.persist()
        logger.info(f"Vector store created and persisted to {persist_dir}")

        return self.vectorstore

    def load_vectorstore(self, persist_dir: str = "./chroma_db"):
        """Load an existing vector store from disk"""
        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            base_url=self.ollama_url,
            model=self.model_name
        )

        # Load existing vector store
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

        logger.info(f"Vector store loaded from {persist_dir}")
        return self.vectorstore

    def setup_retriever(self, search_kwargs: Dict = None):
        """Set up the retriever component"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call setup_vectorstore first.")

        if search_kwargs is None:
            search_kwargs = {
                "k": 5,  # Number of documents to retrieve
                "score_threshold": 0.5  # Minimum relevance score
            }

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        logger.info("Retriever set up successfully")
        return self.retriever

    def setup_llm(self, temperature: float = 0.1):
        """Set up the LLM component"""
        self.llm = ChatOllama(
            base_url=self.ollama_url,
            model=self.model_name,
            temperature=temperature
        )

        logger.info(f"LLM initialized with model {self.model_name}")
        return self.llm

    def create_prompt_template(self):
        """Create the prompt template for the RAG system"""
        template = """
        You are a helpful AI assistant for Marbet, a German event management agency. You are assisting travelers on an incentive trip aboard the Scenic Eclipse ship traveling to Canada and the USA.

        Your name is Marbet Assistant and you specialize in providing information about:
        - Daily schedules and activities
        - Travel requirements and documentation
        - Ship services and policies
        - WiFi and connectivity
        - General travel assistance

        CONTEXT INFORMATION:
        {context}

        USER QUESTION:
        {question}

        GUIDELINES:
        1. Answer based ONLY on the information provided in the context
        2. If the answer is not in the context, politely say you don't have that specific information
        3. Be concise but complete
        4. Maintain a professional, helpful tone
        5. Format your response in a clear, easy-to-read manner
        6. If referring to dates or events, be specific about when they occur

        Your response:
        """

        return PromptTemplate.from_template(template)

    def setup_rag_chain(self):
        """Set up the complete RAG chain"""
        if not self.retriever or not self.llm:
            raise ValueError("Retriever and LLM must be initialized first")

        # Create prompt template
        prompt = self.create_prompt_template()

        # Format retrieved documents
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        # Build the RAG chain
        rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        self.chain = rag_chain
        logger.info("RAG chain setup complete")

        return self.chain

    def setup_conversation_chain(self):
        """Set up a chain with conversation memory"""
        if not self.retriever or not self.llm:
            raise ValueError("Retriever and LLM must be initialized first")

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create prompt template
        prompt = self.create_prompt_template()

        # Create retrieval chain with memory
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "memory": memory
            }
        )

        logger.info("Conversation chain setup complete")
        return self.chain

    def query(self, question: str) -> str:
        """Process a query through the RAG chain"""
        if not self.chain:
            raise ValueError("Chain not initialized. Call setup_rag_chain first.")

        logger.info(f"Processing query: {question}")
        response = self.chain.invoke(question)

        return response

    def initialize_pipeline(self, document_chunks: List = None, persist_dir: str = "./chroma_db"):
        """Initialize the complete pipeline"""
        # Step 1: Set up vector store
        if document_chunks:
            self.setup_vectorstore(document_chunks, persist_dir)
        else:
            try:
                self.load_vectorstore(persist_dir)
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                raise ValueError("Either provide document_chunks or ensure persist_dir exists")

        # Step 2: Set up retriever
        self.setup_retriever()

        # Step 3: Set up LLM
        self.setup_llm()

        # Step 4: Set up RAG chain
        self.setup_rag_chain()

        logger.info("Pipeline initialization complete")
        return self
import os
import pandas as pd
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process event documents for the Marbet chatbot"""

    def __init__(self, document_dir: str = "./docs"):
        self.document_dir = document_dir

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Load all documents from the directory"""
        documents = []

        for filename in os.listdir(self.document_dir):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.document_dir, filename)
                try:
                    # Extract text and metadata
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                    # Add additional metadata
                    for doc in docs:
                        doc.metadata["filename"] = filename
                        doc.metadata["category"] = self._categorize_document(filename)
                        documents.append(doc)

                    print(f"Successfully loaded {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Loaded {len(documents)} document pages in total")
        return documents

    def _categorize_document(self, filename: str) -> str:
        """Categorize documents based on filename"""
        categories = {
            "Activity": "activities",
            "WiFi": "connectivity",
            "A-Z": "ship_services",
            "Eclipse": "ship_services",
            "Pack": "travel_requirements",
            "SPA": "wellness",
            "ESTA": "visa_requirements",
            "eTA": "visa_requirements"
        }

        for key, category in categories.items():
            if key in filename:
                return category

        return "general"

    def split_documents(self, documents: List, chunk_size: int = 1000,
                        chunk_overlap: int = 200) -> List:
        """Split documents into smaller chunks for better retrieval"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")

        # Add additional metadata to help with retrieval
        for i, chunk in enumerate(chunks):
            # Extract date information if available (especially for activities)
            if chunk.metadata.get("category") == "activities":
                dates = self._extract_dates(chunk.page_content)
                if dates:
                    chunk.metadata["dates"] = dates

            # Add chunk ID for reference
            chunk.metadata["chunk_id"] = i

        return chunks

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text (especially for activities)"""
        # Simple pattern matching for dates in format: DD.MM.YYYY
        import re
        date_pattern = r'\d{2}\.\d{2}\.202\d'
        dates = re.findall(date_pattern, text)

        # Also look for date mentions in text
        date_keywords = ["Saturday", "Sunday", "Monday", "Tuesday",
                         "Wednesday", "Thursday", "Friday"]

        for keyword in date_keywords:
            if keyword in text:
                # Find the line containing the date
                lines = text.split('\n')
                for line in lines:
                    if keyword in line:
                        # Extract the full date context
                        if line not in dates:
                            dates.append(line)

        return dates

    def create_structured_data(self, documents: List) -> Dict[str, pd.DataFrame]:
        """Create structured dataframes for specific data types"""
        structured_data = {}

        # Create activities dataframe
        activity_docs = [doc for doc in documents if doc.metadata.get("category") == "activities"]
        if activity_docs:
            activities = []
            for doc in activity_docs:
                # Extract activities from text
                # This could be enhanced with more advanced parsing
                sections = doc.page_content.split('\n\n')
                current_date = None

                for section in sections:
                    if "Election program" in section or "Optional program" in section:
                        # Extract activity name
                        try:
                            name = section.split("Election program: ")[1].split("\n")[0]
                        except:
                            try:
                                name = section.split("Optional program: ")[1].split("\n")[0]
                            except:
                                name = "Unknown activity"

                        activities.append({
                            "date": current_date,
                            "name": name,
                            "description": section,
                            "category": "activity"
                        })
                    elif any(day in section for day in ["Saturday", "Sunday", "Monday", "Tuesday",
                                                        "Wednesday", "Thursday", "Friday"]):
                        # This is likely a date section
                        current_date = section

            structured_data["activities"] = pd.DataFrame(activities)

        # Create ship services dataframe
        ship_docs = [doc for doc in documents if doc.metadata.get("category") == "ship_services"]
        if ship_docs:
            services = []
            for doc in ship_docs:
                # Extract A-Z listings
                lines = doc.page_content.split('\n')
                current_service = None
                current_description = ""

                for line in lines:
                    # Check if this is a new service header (typically single letters or short words)
                    if len(line.strip()) <= 3 and line.strip().isalpha():
                        # Save previous service if it exists
                        if current_service:
                            services.append({
                                "service": current_service,
                                "description": current_description.strip(),
                                "category": "ship_service"
                            })

                        current_service = line.strip()
                        current_description = ""
                    elif current_service:
                        current_description += line + " "

            # Add the last service
            if current_service:
                services.append({
                    "service": current_service,
                    "description": current_description.strip(),
                    "category": "ship_service"
                })

            structured_data["ship_services"] = pd.DataFrame(services)

        return structured_data

    def process_all(self):
        """Run the complete document processing pipeline"""
        # Load all documents
        documents = self.load_all_documents()

        # Create structured data (optional)
        structured_data = self.create_structured_data(documents)

        # Split documents into chunks
        chunks = self.split_documents(documents)

        return {
            "documents": documents,
            "chunks": chunks,
            "structured_data": structured_data
        }
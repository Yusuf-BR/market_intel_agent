import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorLibrarian:
    def __init__(self):
        # Using the model required for your project
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vector_db = None

    def prepare_data(self, text):
        """Splits raw markdown into manageable chunks for the agent."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        return text_splitter.split_text(text)

    def build_index(self, chunks):
        """Creates the FAISS vector database."""
        self.vector_db = FAISS.from_texts(chunks, self.embeddings)

    def save_local(self, folder_path):
        """Saves index.faiss AND index.pkl so the tool works."""
        if self.vector_db:
            os.makedirs(folder_path, exist_ok=True)
            self.vector_db.save_local(folder_path)
            print(f"✅ Database saved to {folder_path}")

    def load_local(self, folder_path):
        """Used by the tool to read the data."""
        self.vector_db = FAISS.load_local(
            folder_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
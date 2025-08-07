import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import config

class VectorStore:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embedding model"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """Load and split PDF documents"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        split_documents = text_splitter.split_documents(documents)
        return split_documents
    
    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
    
    def save_vectorstore(self, path: str = config.VECTORSTORE_PATH):
        """Save vector store to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path: str = config.VECTORSTORE_PATH):
        """Load vector store from disk"""
        try:
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever()
            return True
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return False
    
    def search(self, query: str) -> List[Document]:
        """Search vector store for relevant documents"""
        if self.retriever:
            return self.retriever.get_relevant_documents(query)
        return []
    
    def is_available(self) -> bool:
        """Check if vector store is available"""
        return self.vectorstore is not None
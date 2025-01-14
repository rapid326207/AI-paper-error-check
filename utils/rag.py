from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self, openai_api_key: str):
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                timeout=60,  # Increase timeout
                max_retries=3
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise

        # Simplified text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,  # Smaller chunks for better reliability
            chunk_overlap=300,
            length_function=len,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
        )

    def create_chunks_with_metadata(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        chunks = self.text_splitter.create_documents([text], metadatas=[metadata or {}])
        return [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> FAISS:
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        return vectorstore

    def get_relevant_chunks(self, vectorstore: FAISS, query: str, k: int = 3) -> List[str]:
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

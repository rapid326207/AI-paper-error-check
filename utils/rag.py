from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import logging
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        # Reduce chunk size for embeddings (8192 token limit)
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Using ~4K tokens (16K chars) for embeddings to stay safely under limit
            chunk_size=16000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
        )
        
    def create_chunks_with_metadata(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        chunks = self.text_splitter.create_documents([text], metadatas=[metadata or {}])
        return [{"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]

    def get_embedding(self, text: str) -> List[float]:
        try:
            # Ensure text is within token limit (approx 4 chars per token)
            if len(text) > 32000:  # 8K tokens ≈ 32K chars
                text = text[:32000].rsplit('. ', 1)[0] + '.'
                logger.info(f"Text truncated to {len(text)} characters for embedding")
            
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict:
        try:
            processed_texts = []
            processed_embeddings = []
            processed_metadatas = []
            
            for chunk in chunks:
                text = chunk["text"]
                metadata = chunk["metadata"]
                
                # Ensure text is within token limit (approx 4 chars per token)
                if len(text) > 32000:  # 8K tokens ≈ 32K chars
                    text = text[:32000].rsplit('. ', 1)[0] + '.'
                
                embedding = self.get_embedding(text)
                
                processed_texts.append(text)
                processed_embeddings.append(embedding)
                processed_metadatas.append(metadata)

            return {
                "texts": processed_texts,
                "embeddings": processed_embeddings,
                "metadatas": processed_metadatas
            }

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise Exception(f"Embedding creation failed: {str(e)}")

    def get_relevant_chunks(self, stored_data: Dict, query: str, k: int = 3) -> List[str]:
        try:
            query_embedding = self.get_embedding(query)
            similarities = cosine_similarity(
                [query_embedding],
                stored_data["embeddings"]
            )[0]
            
            # Get top k most similar chunks
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            return [stored_data["texts"][i] for i in top_k_indices]
            
        except Exception as e:
            logger.error(f"Error getting relevant chunks: {e}")
            return []

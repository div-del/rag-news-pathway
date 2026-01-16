"""
Local Embedder for Pathway.
Wraps sentence-transformers to work with Pathway's DocumentStore interface.
"""

import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class LocalSentenceTransformerEmbedder:
    """
    Pathway-compatible embedder using local sentence-transformers.
    
    This embedder runs entirely locally (no API costs) and provides
    high-quality embeddings for semantic search.
    
    Compatible with Pathway's UDF system for streaming operations.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None  # Auto-detect: CUDA if available, else CPU
    ):
        """
        Initialize the local embedder.
        
        Args:
            model_name: HuggingFace model name for sentence-transformers
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._embedding_dimension = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        # Load model lazily to avoid slow startup
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension by encoding a test string
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            self._embedding_dimension = test_embedding.shape[1]
            
            logger.info(
                f"Embedding model loaded: {self.model_name} "
                f"(dim={self._embedding_dimension}, device={self._model.device})"
            )
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        if self._embedding_dimension is None:
            self._load_model()
        return self._embedding_dimension
    
    def __call__(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        This method is called by Pathway's DocumentStore during indexing.
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            List of embedding vectors as lists of floats
        """
        if self._model is None:
            self._load_model()
        
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter out empty strings
        texts = [t if t else " " for t in texts]
        
        try:
            # Generate embeddings
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Convert to list of lists for JSON serialization
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vectors on error
            return [[0.0] * self._embedding_dimension for _ in texts]
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.__call__([query])
        return embeddings[0]
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embedding vectors
        """
        return self.__call__(documents)


# Singleton instance for reuse
_embedder_instance: LocalSentenceTransformerEmbedder = None


def get_local_embedder(model_name: str = "all-MiniLM-L6-v2") -> LocalSentenceTransformerEmbedder:
    """
    Get or create the local embedder singleton.
    
    Args:
        model_name: Model to use (only applies on first call)
        
    Returns:
        LocalSentenceTransformerEmbedder instance
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = LocalSentenceTransformerEmbedder(model_name=model_name)
    return _embedder_instance


if __name__ == "__main__":
    # Test the embedder
    logging.basicConfig(level=logging.INFO)
    
    embedder = LocalSentenceTransformerEmbedder()
    
    # Test single text
    result = embedder(["Hello, world!"])
    print(f"Single text embedding shape: {len(result[0])}")
    
    # Test multiple texts
    texts = [
        "Tesla announces new electric vehicle",
        "Apple releases new iPhone",
        "Stock market reaches all-time high"
    ]
    results = embedder(texts)
    print(f"Multiple texts: {len(results)} embeddings, dim={len(results[0])}")
    
    # Test query embedding
    query_emb = embedder.embed_query("What's new with Tesla?")
    print(f"Query embedding dim: {len(query_emb)}")

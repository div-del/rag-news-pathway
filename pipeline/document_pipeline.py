"""
Pathway Document Pipeline.
Handles real-time document ingestion, embedding, and indexing using Pathway.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

import pathway as pw
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter

from config import Config

logger = logging.getLogger(__name__)


class NewsArticleSchema(pw.Schema):
    """Schema for news articles in Pathway"""
    article_id: str = pw.column_definition(primary_key=True)
    url: str
    title: str
    content: str
    author: Optional[str]
    publish_date: Optional[str]
    source: str
    category: str
    topics: str  # JSON-encoded list
    scraped_at: str


class ArticleChunkSchema(pw.Schema):
    """Schema for chunked articles"""
    chunk_id: str = pw.column_definition(primary_key=True)
    article_id: str
    chunk_index: int
    text: str
    metadata: str  # JSON-encoded metadata


class PathwayDocumentPipeline:
    """
    Pathway-based document processing pipeline.
    Handles:
    - Continuous document ingestion
    - Text chunking
    - Embedding generation
    - Vector indexing
    """
    
    def __init__(
        self,
        openai_api_key: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.openai_api_key = openai_api_key or Config.OPENROUTER_API_KEY
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.chunk_size = chunk_size or Config.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.RAG_CHUNK_OVERLAP
        
        # Initialize embedder
        self._embedder = OpenAIEmbedder(
            api_key=self.openai_api_key,
            model=self.embedding_model,
        )
        
        # Initialize text splitter
        self._splitter = TokenCountSplitter(
            max_tokens=self.chunk_size // 4,  # Rough token estimate
            overlap=self.chunk_overlap // 4
        )
        
        # Document store will be initialized with the pipeline
        self._document_store: Optional[DocumentStore] = None
        self._articles_table: Optional[pw.Table] = None
        
        logger.info(f"Document pipeline initialized with model: {self.embedding_model}")
    
    def create_article_input_connector(self):
        """
        Create a Pathway input connector for articles.
        Uses a Python connector that accepts data programmatically.
        """
        # Create a Python connector that can receive data
        class ArticleInputConnector(pw.io.python.ConnectorSubject):
            def __init__(self):
                super().__init__()
                self._buffer: List[Dict] = []
            
            def add_article(self, article: Dict[str, Any]):
                """Add an article to the stream"""
                topics_json = json.dumps(article.get("topics", []))
                
                data = {
                    "article_id": article.get("article_id", ""),
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "content": article.get("content", ""),
                    "author": article.get("author"),
                    "publish_date": article.get("publish_date"),
                    "source": article.get("source", ""),
                    "category": article.get("category", ""),
                    "topics": topics_json,
                    "scraped_at": article.get("scraped_at", datetime.utcnow().isoformat())
                }
                self.next(**data)
            
            def run(self):
                """Required by ConnectorSubject"""
                # Keep running indefinitely
                import time
                while True:
                    time.sleep(1)
        
        return ArticleInputConnector()
    
    def build_pipeline(self, input_connector) -> pw.Table:
        """
        Build the Pathway document processing pipeline.
        
        Args:
            input_connector: ArticleInputConnector instance
        
        Returns:
            Pathway table with processed documents
        """
        # Create input table from connector
        articles = pw.io.python.read(
            input_connector,
            schema=NewsArticleSchema
        )
        
        self._articles_table = articles
        
        # Transform: Create text for embedding
        articles_with_text = articles.select(
            *articles,
            embedding_text=pw.apply(
                lambda title, content: f"{title}\n\n{content}",
                articles.title,
                articles.content
            ),
            metadata_json=pw.apply(
                lambda article_id, url, category, source, topics: json.dumps({
                    "article_id": article_id,
                    "url": url,
                    "category": category,
                    "source": source,
                    "topics": json.loads(topics) if topics else []
                }),
                articles.article_id,
                articles.url,
                articles.category,
                articles.source,
                articles.topics
            )
        )
        
        logger.info("Built article ingestion pipeline")
        return articles_with_text
    
    def create_document_store(
        self,
        articles_table: pw.Table,
        index_type: str = "hybrid"  # "vector", "text", or "hybrid"
    ) -> DocumentStore:
        """
        Create a Pathway DocumentStore from articles table.
        
        Args:
            articles_table: Processed articles table
            index_type: Type of index to use
        
        Returns:
            DocumentStore instance
        """
        # Create document store with hybrid indexing
        self._document_store = DocumentStore(
            documents=articles_table,
            embedder=self._embedder,
            splitter=self._splitter,
            # Use text column for content
            text_column="embedding_text",
            metadata_column="metadata_json",
        )
        
        logger.info(f"Created DocumentStore with {index_type} indexing")
        return self._document_store
    
    def get_document_store(self) -> Optional[DocumentStore]:
        """Get the document store instance"""
        return self._document_store
    
    def query_documents(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: str = None
    ) -> pw.Table:
        """
        Query the document store.
        
        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Optional metadata filter expression
        
        Returns:
            Table of matching documents
        """
        if self._document_store is None:
            raise RuntimeError("Document store not initialized")
        
        top_k = top_k or Config.RAG_TOP_K
        
        return self._document_store.query(
            query=query,
            k=top_k,
            metadata_filter=metadata_filter
        )


def create_full_pipeline():
    """
    Create the complete document processing pipeline.
    Returns the pipeline object and input connector.
    """
    pipeline = PathwayDocumentPipeline()
    input_connector = pipeline.create_article_input_connector()
    articles_table = pipeline.build_pipeline(input_connector)
    document_store = pipeline.create_document_store(articles_table)
    
    return {
        "pipeline": pipeline,
        "input_connector": input_connector,
        "articles_table": articles_table,
        "document_store": document_store
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test pipeline creation
    print("Creating Pathway document pipeline...")
    components = create_full_pipeline()
    print("Pipeline created successfully!")
    print(f"Components: {list(components.keys())}")

"""
RAG Engine with Context Expansion.
Handles queries with global, article-specific, and comparison contexts.
Now with vector similarity search using local embeddings.

Supports two modes:
1. Local RAG: In-memory embeddings with sentence-transformers
2. Pathway RAG: Real-time streaming with Pathway DocumentStore
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
from openai import AsyncOpenAI

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

from config import Config

logger = logging.getLogger(__name__)

# Pathway server reference (set by app.py on startup)
_pathway_server = None


def set_pathway_server(server):
    """Set the Pathway server reference for RAG queries"""
    global _pathway_server
    _pathway_server = server
    logger.info("Pathway server connected to RAG engine")


def get_pathway_server():
    """Get the Pathway server reference"""
    global _pathway_server
    return _pathway_server


@dataclass 
class RAGContext:
    """Represents retrieved context for RAG"""
    documents: List[Dict[str, Any]]
    query: str
    context_type: str  # "global", "article", "comparison"
    relevance_scores: List[float] = field(default_factory=list)
    metadata_filter: Optional[str] = None
    total_tokens: int = 0
    search_method: str = "hybrid"  # "vector", "keyword", "hybrid", or "pathway"
    search_latency_ms: float = 0.0  # Time taken for search
    pathway_used: bool = False  # Whether Pathway was used


@dataclass
class RAGResponse:
    """Response from RAG query"""
    query: str
    response: str
    context: RAGContext
    model: str
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        docs_with_scores = []
        for i, d in enumerate(self.context.documents):
            score = self.context.relevance_scores[i] if i < len(self.context.relevance_scores) else 0.0
            docs_with_scores.append({
                "article_id": d.get("article_id"),
                "title": d.get("title"),
                "snippet": d.get("content", d.get("text", ""))[:200],
                "source": d.get("source", "Unknown"),
                "score": round(score, 3)
            })
        result["context"]["documents"] = docs_with_scores
        return result


class RAGEngine:
    """
    RAG Engine with multiple context modes:
    - Global: Search all indexed articles
    - Article-specific: Focus on one article + related topics  
    - Comparison: Merge context from multiple articles
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL
        
        # Initialize OpenAI client (works with OpenRouter)
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # In-memory document store (backed by SQLite)
        self._documents: Dict[str, Dict[str, Any]] = {}
        
        # Vector embeddings store
        self._embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize local embedding model (runs on CPU, no API cost)
        self._embedder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {embedding_model}...")
                self._embedder = SentenceTransformer(embedding_model)
                logger.info(f"Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self._embedder = None
        else:
            logger.warning("sentence-transformers not installed. Using keyword search only.")
        
        # Load existing articles from SQLite
        self._load_from_store()
        
        # Generate embeddings for loaded articles
        if self._embedder and self._documents:
            self._generate_embeddings_batch(list(self._documents.values()))
        
        logger.info(f"RAG Engine initialized with model: {self.model}, loaded {len(self._documents)} articles, {len(self._embeddings)} embeddings")
    
    def _load_from_store(self):
        """Load all articles from SQLite into memory"""
        try:
            from api.article_store import get_article_store
            store = get_article_store()
            articles = store.get_all_articles()
            for article in articles:
                article_id = article.get("article_id")
                if article_id:
                    self._documents[article_id] = article
            logger.info(f"Loaded {len(articles)} articles from SQLite")
        except Exception as e:
            logger.warning(f"Could not load articles from store: {e}")
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """Add a document to the in-memory store, generate embedding, and persist to SQLite"""
        article_id = document.get("article_id")
        if article_id:
            self._documents[article_id] = document
            
            # Generate embedding for this document
            if self._embedder:
                self._generate_embedding(document)
            
            logger.debug(f"Added document with embedding: {article_id}")
            
            # Persist to SQLite
            try:
                from api.article_store import get_article_store
                store = get_article_store()
                return store.add_article(document)
            except Exception as e:
                logger.warning(f"Could not persist to store: {e}")
                return True  # Still added to memory
        return False
    
    def _generate_embedding(self, document: Dict[str, Any]) -> None:
        """Generate embedding for a single document"""
        if not self._embedder:
            return
        
        article_id = document.get("article_id")
        if not article_id:
            return
        
        # Create text for embedding (title + content + topics)
        text = self._get_document_text(document)
        
        try:
            embedding = self._embedder.encode(text, convert_to_numpy=True)
            self._embeddings[article_id] = embedding
        except Exception as e:
            logger.warning(f"Could not generate embedding for {article_id}: {e}")
    
    def _generate_embeddings_batch(self, documents: List[Dict[str, Any]]) -> None:
        """Generate embeddings for multiple documents efficiently"""
        if not self._embedder or not documents:
            return
        
        texts = []
        article_ids = []
        
        for doc in documents:
            article_id = doc.get("article_id")
            if article_id and article_id not in self._embeddings:
                texts.append(self._get_document_text(doc))
                article_ids.append(article_id)
        
        if not texts:
            return
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
            for article_id, embedding in zip(article_ids, embeddings):
                self._embeddings[article_id] = embedding
            
            logger.info(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.warning(f"Batch embedding generation failed: {e}")
    
    def _get_document_text(self, document: Dict[str, Any]) -> str:
        """Get searchable text from a document"""
        title = document.get("title", "")
        content = document.get("content", "")[:2000]  # Limit content length for embedding
        topics = document.get("topics", [])
        if isinstance(topics, str):
            try:
                topics = json.loads(topics)
            except:
                topics = []
        topics_str = " ".join(topics) if topics else ""
        
        return f"{title}\n{content}\n{topics_str}"
    
    def _vector_search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        article_ids: Optional[List[str]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Vector similarity search using cosine similarity.
        Returns list of (document, score) tuples.
        """
        if not self._embedder or not self._embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self._embedder.encode(query, convert_to_numpy=True)
        
        # Calculate cosine similarity with all document embeddings
        scored_docs = []
        
        for article_id, doc_embedding in self._embeddings.items():
            # Apply article ID filter if specified
            if article_ids and article_id not in article_ids:
                continue
            
            doc = self._documents.get(article_id)
            if not doc:
                continue
            
            # Apply category filter
            if category_filter:
                doc_category = doc.get("category", "").lower()
                if category_filter.lower() not in doc_category:
                    continue
            
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8
            )
            
            scored_docs.append((doc, float(similarity)))
        
        # Sort by similarity score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        article_ids: Optional[List[str]] = None,
        vector_weight: float = 0.7
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Hybrid search combining vector and keyword search.
        Returns list of (document, score) tuples.
        """
        # Get results from both methods
        vector_results = self._vector_search(query, top_k * 2, category_filter, article_ids)
        keyword_results = self._simple_search(query, top_k * 2, category_filter, article_ids)
        
        # Normalize keyword scores to 0-1 range
        max_keyword_score = max((s for _, s in keyword_results), default=1) if keyword_results else 1
        
        # Combine scores using weighted fusion
        combined_scores: Dict[str, Tuple[Dict[str, Any], float]] = {}
        
        for doc, score in vector_results:
            article_id = doc.get("article_id")
            if article_id:
                combined_scores[article_id] = (doc, score * vector_weight)
        
        for doc, score in keyword_results:
            article_id = doc.get("article_id")
            if article_id:
                normalized_score = (score / max_keyword_score) if max_keyword_score > 0 else 0
                if article_id in combined_scores:
                    existing_doc, existing_score = combined_scores[article_id]
                    combined_scores[article_id] = (existing_doc, existing_score + normalized_score * (1 - vector_weight))
                else:
                    combined_scores[article_id] = (doc, normalized_score * (1 - vector_weight))
        
        # Sort by combined score
        results = list(combined_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _pathway_search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> Tuple[List[Tuple[Dict[str, Any], float]], float]:
        """
        Search using Pathway's DocumentStore.
        Returns (results, latency_ms).
        
        This leverages Pathway's real-time incremental indexing.
        """
        pathway_server = get_pathway_server()
        if not pathway_server or not pathway_server.is_running:
            return [], 0.0
        
        start_time = time.time()
        
        try:
            # Query Pathway server
            results = pathway_server.query(
                query=query,
                top_k=top_k,
                metadata_filter=category_filter
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Convert to (document, score) tuples
            scored_docs = []
            for item in results:
                if isinstance(item, dict):
                    score = item.get("score", 0.5)
                    doc = {
                        "article_id": item.get("article_id", ""),
                        "title": item.get("title", ""),
                        "content": item.get("text", item.get("content", "")),
                        "source": item.get("source", "Unknown"),
                        "category": item.get("category", ""),
                        "topics": item.get("topics", []),
                        "url": item.get("url", ""),
                        "publish_date": item.get("publish_date", "")
                    }
                    scored_docs.append((doc, score))
            
            logger.debug(f"Pathway search returned {len(scored_docs)} results in {latency_ms:.1f}ms")
            return scored_docs, latency_ms
            
        except Exception as e:
            logger.error(f"Pathway search error: {e}")
            return [], 0.0

    
    def _simple_search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        article_ids: Optional[List[str]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Keyword-based search for fallback and hybrid search.
        Returns list of (document, score) tuples.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        scored_docs = []
        
        for article_id, doc in self._documents.items():
            # Apply article ID filter if specified
            if article_ids and article_id not in article_ids:
                continue
            
            # Apply category filter
            if category_filter:
                doc_category = doc.get("category", "").lower()
                if category_filter.lower() not in doc_category:
                    continue
            
            # Simple term matching score
            text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
            topics = doc.get("topics", [])
            if isinstance(topics, str):
                try:
                    topics = json.loads(topics) if topics else []
                except:
                    topics = []
            
            score = sum(1 for term in query_terms if term in text)
            score += sum(2 for term in query_terms if any(term in t.lower() for t in topics))
            
            if score > 0:
                scored_docs.append((doc, float(score)))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
    
    def _build_context_text(self, documents: List[Dict[str, Any]]) -> str:
        """Build context text from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")[:2000]  # Limit content length
            source = doc.get("source", "Unknown")
            category = doc.get("category", "")
            
            context_parts.append(f"""
Article {i}: {title}
Source: {source} | Category: {category}
---
{content}
---
""")
        
        return "\n".join(context_parts)
    
    async def query_global(
        self,
        query: str,
        top_k: int = None,
        category: Optional[str] = None,
        use_hybrid: bool = True,
        use_pathway: bool = True
    ) -> RAGResponse:
        """
        Query with global context (all articles).
        Uses Pathway when available, falls back to hybrid search.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            category: Optional category filter
            use_hybrid: Whether to use hybrid search (default: True)
            use_pathway: Whether to try Pathway first (default: True)
        
        Returns:
            RAGResponse with answer and context
        """
        top_k = top_k or Config.RAG_TOP_K
        
        search_results = []
        search_method = "keyword"
        search_latency_ms = 0.0
        pathway_used = False
        
        # Try Pathway first if enabled and available
        if use_pathway and Config.USE_PATHWAY:
            pathway_results, latency = self._pathway_search(
                query=query,
                top_k=top_k,
                category_filter=category
            )
            if pathway_results:
                search_results = pathway_results
                search_method = "pathway"
                search_latency_ms = latency
                pathway_used = True
                logger.info(f"Pathway search returned {len(search_results)} results in {latency:.1f}ms")
        
        # Fall back to local search if Pathway didn't return results
        if not search_results:
            start_time = time.time()
            
            if use_hybrid and self._embedder and self._embeddings:
                search_results = self._hybrid_search(
                    query=query,
                    top_k=top_k,
                    category_filter=category
                )
                search_method = "hybrid"
            else:
                search_results = self._simple_search(
                    query=query,
                    top_k=top_k,
                    category_filter=category
                )
                search_method = "keyword"
            
            search_latency_ms = (time.time() - start_time) * 1000
        
        # Extract documents and scores
        documents = [doc for doc, _ in search_results]
        relevance_scores = [score for _, score in search_results]
        
        context = RAGContext(
            documents=documents,
            query=query,
            context_type="global",
            relevance_scores=relevance_scores,
            metadata_filter=f"category={category}" if category else None,
            search_method=search_method,
            search_latency_ms=search_latency_ms,
            pathway_used=pathway_used
        )
        
        if not documents:
            return RAGResponse(
                query=query,
                response="I don't have any relevant news articles to answer this question. Please try a different query or wait for more news to be indexed.",
                context=context,
                model=self.model,
                created_at=datetime.utcnow().isoformat()
            )
        
        # Build prompt
        context_text = self._build_context_text(documents)
        
        system_prompt = """You are a knowledgeable AI news analyst. Your role is to answer questions about current news based on the provided article context.

Guidelines:
- Base your answers primarily on the provided articles
- If the articles don't contain enough information, say so
- Mention specific sources when citing information
- Be concise but informative
- Highlight key points and developments"""

        user_prompt = f"""Based on the following news articles, please answer this question:

**Question:** {query}

**Context (Retrieved Articles):**
{context_text}

Please provide a comprehensive answer based on the articles above."""

        # Generate response
        response = await self._generate_response(system_prompt, user_prompt)
        
        return RAGResponse(
            query=query,
            response=response,
            context=context,
            model=self.model,
            created_at=datetime.utcnow().isoformat()
        )
    
    async def query_article(
        self,
        query: str,
        article_id: str,
        expand_context: bool = True,
        top_k: int = 3
    ) -> RAGResponse:
        """
        Query with article-specific context.
        Optionally expands to related articles based on topics.
        
        Args:
            query: User's question  
            article_id: ID of the main article
            expand_context: Whether to include related articles
            top_k: Number of related articles to include
        
        Returns:
            RAGResponse with answer and context
        """
        # Get the main article
        main_article = self._documents.get(article_id)
        
        if not main_article:
            return RAGResponse(
                query=query,
                response=f"Article with ID '{article_id}' not found.",
                context=RAGContext(documents=[], query=query, context_type="article"),
                model=self.model,
                created_at=datetime.utcnow().isoformat()
            )
        
        documents = [main_article]
        
        # Expand context with related articles
        if expand_context:
            topics = main_article.get("topics", [])
            if isinstance(topics, str):
                topics = json.loads(topics) if topics else []
            
            if topics:
                # Search for related articles by topics
                topic_query = " ".join(topics[:3])  # Use top 3 topics
                related = self._simple_search(
                    query=topic_query,
                    top_k=top_k + 1  # +1 to account for main article
                )
                
                # Add related articles (excluding main)
                # _simple_search returns (doc, score) tuples
                for doc, score in related:
                    if doc.get("article_id") != article_id:
                        documents.append(doc)
                        if len(documents) > top_k + 1:
                            break
        
        context = RAGContext(
            documents=documents,
            query=query,
            context_type="article",
            metadata_filter=f"article_id={article_id}"
        )
        
        # Build prompt with emphasis on main article
        main_context = self._build_context_text([main_article])
        related_context = self._build_context_text(documents[1:]) if len(documents) > 1 else ""
        
        system_prompt = """You are an AI assistant helping users understand news articles. Focus primarily on the main article but reference related articles when relevant.

Guidelines:  
- Answer based on the main article first
- Use related articles to provide additional context
- If the user asks about something not in the articles, say so
- Be conversational and helpful"""

        user_prompt = f"""The user is reading the following news article and has a question:

**MAIN ARTICLE:**
{main_context}

**RELATED ARTICLES (for context):**
{related_context if related_context else "No related articles available."}

**User's Question:** {query}

Please answer based on the articles above, with emphasis on the main article."""

        response = await self._generate_response(system_prompt, user_prompt)
        
        return RAGResponse(
            query=query,
            response=response,
            context=context,
            model=self.model,
            created_at=datetime.utcnow().isoformat()
        )
    
    async def query_comparison(
        self,
        query: str,
        article_ids: List[str]
    ) -> RAGResponse:
        """
        Query with comparison context (multiple articles merged).
        Perfect for "compare Tesla vs BMW" type queries.
        
        Args:
            query: User's comparison question
            article_ids: IDs of articles to compare
        
        Returns:
            RAGResponse with comparison analysis
        """
        # Get all specified articles
        documents = []
        missing_ids = []
        
        for article_id in article_ids:
            article = self._documents.get(article_id)
            if article:
                documents.append(article)
            else:
                missing_ids.append(article_id)
        
        if not documents:
            return RAGResponse(
                query=query,
                response="None of the specified articles were found.",
                context=RAGContext(documents=[], query=query, context_type="comparison"),
                model=self.model,
                created_at=datetime.utcnow().isoformat()
            )
        
        context = RAGContext(
            documents=documents,
            query=query,
            context_type="comparison",
            metadata_filter=f"article_ids={','.join(article_ids)}"
        )
        
        # Build comparison prompt
        articles_context = ""
        for i, doc in enumerate(documents, 1):
            articles_context += f"""
**ARTICLE {i}: {doc.get('title', 'Untitled')}**
Source: {doc.get('source', 'Unknown')} | Category: {doc.get('category', '')}
Published: {doc.get('publish_date', 'Unknown date')}

{doc.get('content', '')[:2500]}

---
"""
        
        system_prompt = """You are an AI analyst specializing in comparative news analysis. Your task is to compare and contrast information from multiple news articles.

Guidelines:
- Identify key similarities and differences
- Highlight important points from each article
- Provide balanced analysis
- Use specific details and quotes when relevant
- Structure your comparison clearly"""

        user_prompt = f"""Please analyze and compare the following articles:

{articles_context}

**Comparison Request:** {query}

Provide a detailed comparison based on the articles above."""

        if missing_ids:
            user_prompt += f"\n\nNote: The following article IDs were not found: {', '.join(missing_ids)}"
        
        response = await self._generate_response(system_prompt, user_prompt)
        
        return RAGResponse(
            query=query,
            response=response,
            context=context,
            model=self.model,
            created_at=datetime.utcnow().isoformat()
        )
    
    async def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using LLM"""
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Sorry, I encountered an error generating a response: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics including Pathway status"""
        # Get Pathway stats if available
        pathway_stats = {}
        pathway_server = get_pathway_server()
        if pathway_server:
            pathway_stats = pathway_server.get_stats()
        
        return {
            "indexed_documents": len(self._documents),
            "embeddings_count": len(self._embeddings),
            "embeddings_available": EMBEDDINGS_AVAILABLE and self._embedder is not None,
            "search_method": "hybrid" if (self._embedder and self._embeddings) else "keyword",
            "model": self.model,
            "base_url": self.base_url,
            "pathway_enabled": Config.USE_PATHWAY,
            "pathway_running": pathway_server.is_running if pathway_server else False,
            "pathway_stats": pathway_stats
        }


# Singleton instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine instance"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


async def test_rag_engine():
    """Test the RAG engine"""
    engine = RAGEngine()
    
    # Add test documents
    engine.add_document({
        "article_id": "tesla_001",
        "title": "Tesla Announces New Model Y Refresh",
        "content": "Tesla has announced a major refresh of its popular Model Y electric vehicle. The new version features improved range of 350 miles, faster charging, and updated interior. CEO Elon Musk revealed the changes at a special event.",
        "source": "TechNews",
        "category": "Technology",
        "topics": ["Tesla", "EV", "electric vehicle", "Model Y"]
    })
    
    engine.add_document({
        "article_id": "bmw_001", 
        "title": "BMW Launches New Electric IX Series",
        "content": "BMW has launched its new iX electric SUV series with advanced autonomous driving features. The vehicle offers 300 miles of range and luxury interior. BMW aims to compete directly with Tesla's Model X.",
        "source": "AutoWorld",
        "category": "Technology",
        "topics": ["BMW", "EV", "electric vehicle", "iX", "autonomous"]
    })
    
    print("Testing RAG Engine...")
    
    # Test global query
    result = await engine.query_global("What's new in electric vehicles?")
    print(f"\nGlobal Query Response:\n{result.response[:500]}...")
    
    # Test comparison
    result = await engine.query_comparison(
        "Compare Tesla and BMW's new electric vehicles",
        ["tesla_001", "bmw_001"]
    )
    print(f"\nComparison Response:\n{result.response[:500]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_rag_engine())

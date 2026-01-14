"""
RAG Engine with Context Expansion.
Handles queries with global, article-specific, and comparison contexts.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI

from config import Config

logger = logging.getLogger(__name__)


@dataclass 
class RAGContext:
    """Represents retrieved context for RAG"""
    documents: List[Dict[str, Any]]
    query: str
    context_type: str  # "global", "article", "comparison"
    metadata_filter: Optional[str] = None
    total_tokens: int = 0


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
        result["context"]["documents"] = [
            {"article_id": d.get("article_id"), "title": d.get("title"), "snippet": d.get("text", "")[:200]}
            for d in self.context.documents
        ]
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
        model: str = None
    ):
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL
        
        # Initialize OpenAI client (works with OpenRouter)
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # In-memory document store (before Pathway integration)
        self._documents: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"RAG Engine initialized with model: {self.model}")
    
    def add_document(self, document: Dict[str, Any]):
        """Add a document to the in-memory store"""
        article_id = document.get("article_id")
        if article_id:
            self._documents[article_id] = document
            logger.debug(f"Added document: {article_id}")
    
    def _simple_search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        article_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search (placeholder for vector search).
        Will be replaced with Pathway DocumentStore query.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        scored_docs = []
        
        for article_id, doc in self._documents.items():
            # Apply article ID filter if specified
            if article_ids and article_id not in article_ids:
                continue
            
            # Apply category filter
            if category_filter and doc.get("category", "").lower() != category_filter.lower():
                continue
            
            # Simple term matching score
            text = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
            topics = doc.get("topics", [])
            if isinstance(topics, str):
                topics = json.loads(topics) if topics else []
            
            score = sum(1 for term in query_terms if term in text)
            score += sum(2 for term in query_terms if any(term in t.lower() for t in topics))
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
    
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
        category: Optional[str] = None
    ) -> RAGResponse:
        """
        Query with global context (all articles).
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            category: Optional category filter
        
        Returns:
            RAGResponse with answer and context
        """
        top_k = top_k or Config.RAG_TOP_K
        
        # Retrieve relevant documents
        documents = self._simple_search(
            query=query,
            top_k=top_k,
            category_filter=category
        )
        
        context = RAGContext(
            documents=documents,
            query=query,
            context_type="global",
            metadata_filter=f"category={category}" if category else None
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
                for doc in related:
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
        """Get engine statistics"""
        return {
            "indexed_documents": len(self._documents),
            "model": self.model,
            "base_url": self.base_url
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

"""
AI Chat Engine - Intelligent chat with news article context.
Uses efficient two-tier retrieval: local search first, then fetch new if needed.
Now with semantic relevance scoring and multi-topic search.
Maintains conversation history for contextual responses.
"""

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from openai import AsyncOpenAI
import numpy as np

from config import Config
from api.article_store import get_article_store

# Try to import embedding model for relevance scoring
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    sources: Optional[List[Dict[str, Any]]] = None


@dataclass
class ChatResponse:
    """Response from AI Chat"""
    response: str
    sources: List[Dict[str, Any]]
    articles_searched: int
    articles_fetched: int
    session_id: str
    fetched_topics: List[str] = field(default_factory=list)  # Topics searched for
    relevance_score: float = 0.0  # How relevant the articles are to the query
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AIChatEngine:
    """
    AI Chat Engine with intelligent article context retrieval.
    
    Features:
    - Two-tier search: local SQLite first, then Serper API
    - Semantic relevance scoring with embeddings
    - Multi-topic search for better coverage
    - Topic extraction from user queries
    - Conversation history management
    - Markdown-formatted responses
    - OpenRouter Mistral integration
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        # Use Mistral free model via OpenRouter
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.LLM_BASE_URL
        self.model = "mistralai/mistral-7b-instruct:free"
        
        # Initialize OpenAI client (works with OpenRouter)
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Session storage (in-memory with SQLite backing)
        self._sessions: Dict[str, List[ChatMessage]] = {}
        
        # Initialize embedding model for relevance scoring
        self._embedder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model for relevance scoring: {embedding_model}...")
                self._embedder = SentenceTransformer(embedding_model)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                self._embedder = None
        else:
            logger.warning("sentence-transformers not available for relevance scoring")
        
        # Load sessions from database
        self._load_sessions()
        
        logger.info(f"AI Chat Engine initialized with model: {self.model}")
    
    def _load_sessions(self):
        """Load existing chat sessions from SQLite"""
        try:
            from api.db_models import ChatSession
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from pathlib import Path
            
            db_path = Path(__file__).parent.parent / "articles.db"
            engine = create_engine(f"sqlite:///{db_path}", echo=False)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            chat_sessions = session.query(ChatSession).all()
            for cs in chat_sessions:
                if cs.messages:
                    self._sessions[cs.session_id] = [
                        ChatMessage(**msg) for msg in cs.messages
                    ]
            
            session.close()
            logger.info(f"Loaded {len(self._sessions)} chat sessions")
        except Exception as e:
            logger.warning(f"Could not load chat sessions: {e}")
    
    def _save_session(self, session_id: str):
        """Save a chat session to SQLite"""
        try:
            from api.db_models import ChatSession, Base
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from pathlib import Path
            
            db_path = Path(__file__).parent.parent / "articles.db"
            engine = create_engine(f"sqlite:///{db_path}", echo=False)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            messages = self._sessions.get(session_id, [])
            messages_data = [asdict(m) for m in messages]
            
            # Check if session exists
            existing = session.query(ChatSession).filter_by(session_id=session_id).first()
            
            if existing:
                existing.messages = messages_data
                existing.updated_at = datetime.utcnow()
            else:
                new_session = ChatSession(
                    session_id=session_id,
                    messages=messages_data
                )
                session.add(new_session)
            
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Error saving chat session: {e}")
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        self._save_session(session_id)
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        messages = self._sessions.get(session_id, [])
        return [asdict(m) for m in messages]
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a chat session"""
        if session_id in self._sessions:
            self._sessions[session_id] = []
            self._save_session(session_id)
            return True
        return False
    
    def _extract_topics(self, query: str) -> List[str]:
        """
        Extract key topics from a user query for multi-topic search.
        Returns a list of search queries to try.
        """
        topics = []
        
        # Start with the original query
        topics.append(query)
        
        # Common topic patterns and their expansions
        topic_expansions = {
            # Immigration/Visa related
            "visa": ["US visa policy", "immigration news", "USCIS update"],
            "immigration": ["US immigration news", "visa policy update", "border news"],
            "h1b": ["H1B visa news", "work visa USA", "tech immigration"],
            "green card": ["green card news", "permanent residency USA", "immigration update"],
            
            # Finance related
            "stock": ["stock market news", "trading update", "Wall Street"],
            "finance": ["financial news USA", "economy update", "market report"],
            "market": ["stock market today", "financial markets", "trading news"],
            "bitcoin": ["cryptocurrency news", "bitcoin price", "crypto market"],
            "crypto": ["cryptocurrency news", "bitcoin news", "blockchain"],
            
            # Tech related
            "ai": ["artificial intelligence news", "AI technology", "machine learning"],
            "tech": ["technology news", "Silicon Valley", "tech industry"],
            "tesla": ["Tesla news", "electric vehicle", "Elon Musk"],
            "apple": ["Apple news", "iPhone", "Apple company"],
            "google": ["Google news", "tech giant", "Alphabet"],
            
            # General news
            "economy": ["US economy news", "economic update", "GDP report"],
            "jobs": ["employment news", "job market", "labor market"],
        }
        
        # Extract key terms and find expansions
        query_lower = query.lower()
        for keyword, expansions in topic_expansions.items():
            if keyword in query_lower:
                topics.extend(expansions[:2])  # Add top 2 expansions
        
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for t in topics:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique_topics.append(t)
        
        return unique_topics[:5]  # Limit to 5 topics
    
    def _calculate_relevance(self, query: str, articles: List[Dict[str, Any]]) -> Tuple[float, List[float]]:
        """
        Calculate semantic relevance between query and articles using embeddings.
        Returns (max_score, list of individual scores).
        """
        if not self._embedder or not articles:
            return 0.0, []
        
        try:
            # Encode the query
            query_embedding = self._embedder.encode(query, convert_to_numpy=True)
            
            scores = []
            for article in articles:
                # Build article text for embedding
                title = article.get("title", "")
                content = article.get("content", "")[:1000]
                article_text = f"{title}\n{content}"
                
                # Encode article
                article_embedding = self._embedder.encode(article_text, convert_to_numpy=True)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, article_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding) + 1e-8
                )
                scores.append(float(similarity))
            
            max_score = max(scores) if scores else 0.0
            return max_score, scores
            
        except Exception as e:
            logger.warning(f"Could not calculate relevance: {e}")
            return 0.0, []
    
    def _is_context_relevant(
        self, 
        query: str, 
        articles: List[Dict[str, Any]], 
        threshold: float = 0.35
    ) -> Tuple[bool, float]:
        """
        Determine if retrieved articles are relevant enough to answer the query.
        Returns (is_relevant, max_relevance_score).
        """
        if not articles:
            return False, 0.0
        
        max_score, scores = self._calculate_relevance(query, articles)
        
        # Also check keyword overlap as fallback
        query_terms = set(query.lower().split())
        keyword_matches = 0
        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')[:500]}".lower()
            keyword_matches += sum(1 for term in query_terms if term in text and len(term) > 3)
        
        # Consider relevant if either embedding score or keyword matches are sufficient
        has_good_embedding_score = max_score >= threshold
        has_good_keyword_match = keyword_matches >= len(query_terms) * 0.5
        
        is_relevant = has_good_embedding_score or has_good_keyword_match
        
        logger.info(f"Relevance check: max_score={max_score:.3f}, keyword_matches={keyword_matches}, is_relevant={is_relevant}")
        
        return is_relevant, max_score

    
    def _search_local_articles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search articles in local SQLite store.
        Uses keyword matching for efficiency.
        """
        store = get_article_store()
        
        # Use the store's search method
        results = store.search_articles(query, limit=limit)
        
        if not results:
            # Fallback: try to get recent articles and filter
            all_articles = store.get_all_articles(limit=50)
            query_terms = query.lower().split()
            
            scored = []
            for article in all_articles:
                text = f"{article.get('title', '')} {article.get('content', '')[:500]}".lower()
                topics = article.get('topics', [])
                if isinstance(topics, str):
                    try:
                        topics = json.loads(topics) if topics else []
                    except:
                        topics = []
                
                score = sum(1 for term in query_terms if term in text)
                score += sum(2 for term in query_terms if any(term in t.lower() for t in topics))
                
                if score > 0:
                    scored.append((score, article))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [a for _, a in scored[:limit]]
        
        return results
    
    async def _fetch_new_articles(
        self, 
        query: str, 
        topics: Optional[List[str]] = None,
        num_per_topic: int = 3
    ) -> Tuple[int, List[str]]:
        """
        Fetch new articles from Serper API using multi-topic search.
        Returns (number of articles fetched, list of topics searched).
        """
        try:
            from connectors.news_connector import SerperNewsConnector
            from connectors.article_scraper import ArticleScraper
            from rag.rag_engine import get_rag_engine
            
            connector = SerperNewsConnector()
            scraper = ArticleScraper()
            store = get_article_store()
            rag = get_rag_engine()
            
            # Use provided topics or extract from query
            search_topics = topics if topics else self._extract_topics(query)
            
            logger.info(f"Fetching articles for topics: {search_topics}")
            
            count = 0
            all_urls_seen = set()
            
            for topic in search_topics:
                try:
                    # Search for news on this topic
                    results = await connector.search_news(topic, num_results=num_per_topic)
                    
                    if not results:
                        continue
                    
                    for r in results:
                        # Skip duplicates
                        if r.url in all_urls_seen:
                            continue
                        all_urls_seen.add(r.url)
                        
                        try:
                            # Check if already exists
                            if store.article_exists(r.url):
                                continue
                            
                            # Scrape the article with snippet fallback
                            article = await scraper.scrape_article(
                                url=r.url,
                                category=r.category,
                                source=r.source,
                                snippet=r.snippet  # Use snippet as fallback
                            )
                            
                            if article:
                                article_dict = article.to_dict()
                                
                                # Add to SQLite store
                                if store.add_article(article_dict):
                                    # Also add to RAG engine for embedding generation
                                    rag.add_document(article_dict)
                                    count += 1
                                    logger.debug(f"Indexed: {article.title[:50]}...")
                                    
                        except Exception as e:
                            logger.debug(f"Error fetching article {r.url}: {e}")
                            continue
                    
                    # Small delay between topics
                    await asyncio.sleep(0.3)
                    
                except Exception as e:
                    logger.warning(f"Error searching topic '{topic}': {e}")
                    continue
            
            logger.info(f"Fetched {count} new articles across {len(search_topics)} topics")
            return count, search_topics
            
        except Exception as e:
            logger.error(f"Error fetching new articles: {e}")
            return 0, []
    
    def _build_context(self, articles: List[Dict[str, Any]]) -> str:
        """Build context text from articles for the LLM"""
        if not articles:
            return "No relevant articles found in the database."
        
        context_parts = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "Untitled")
            content = article.get("content", "")[:1500]  # Limit content
            source = article.get("source", "Unknown")
            category = article.get("category", "")
            date = article.get("publish_date", "")
            
            context_parts.append(f"""
**Article {i}: {title}**
Source: {source} | Category: {category} | Date: {date}
---
{content}
---
""")
        
        return "\n".join(context_parts)
    
    def _build_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Build conversation history for LLM context"""
        messages = self._sessions.get(session_id, [])
        
        # Take last N messages
        recent = messages[-limit:] if len(messages) > limit else messages
        
        history = []
        for msg in recent:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return history
    
    async def chat(
        self,
        session_id: str,
        message: str,
        user_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Process a chat message and return AI response.
        
        Enhanced with:
        - Semantic relevance scoring to detect if articles answer the query
        - Multi-topic search for better coverage
        - Better feedback when no relevant articles found
        
        Args:
            session_id: Chat session ID
            message: User's message
            user_id: Optional user ID for personalization
        
        Returns:
            ChatResponse with AI response and sources
        """
        # Ensure session exists
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        # Step 1: Extract topics from query for potential multi-topic search
        extracted_topics = self._extract_topics(message)
        fetched_topics = []
        
        # Step 2: Search local articles
        local_articles = self._search_local_articles(message, limit=5)
        articles_searched = len(local_articles)
        articles_fetched = 0
        
        # Step 3: Check if local articles are actually relevant
        is_relevant, relevance_score = self._is_context_relevant(message, local_articles)
        
        # Step 4: If not relevant or insufficient, fetch new articles
        if not is_relevant or len(local_articles) < 2:
            logger.info(f"Context not relevant enough (score={relevance_score:.3f}), fetching new articles...")
            
            # Use multi-topic search for better coverage
            articles_fetched, fetched_topics = await self._fetch_new_articles(
                query=message,
                topics=extracted_topics,
                num_per_topic=3
            )
            
            if articles_fetched > 0:
                # Re-search to include new articles
                local_articles = self._search_local_articles(message, limit=5)
                articles_searched = len(local_articles)
                
                # Re-check relevance
                is_relevant, relevance_score = self._is_context_relevant(message, local_articles)
        
        # Step 5: Build context
        article_context = self._build_context(local_articles)
        
        # Step 6: Build conversation history
        history = self._build_conversation_history(session_id)
        
        # Step 7: Create system prompt (enhanced based on relevance)
        if is_relevant:
            system_prompt = """You are an intelligent AI news assistant. Your role is to help users understand and explore news topics using the provided article context.

## Response Guidelines:
1. **Use Markdown Formatting** - Structure your responses with headers, bullet points, and bold text for readability
2. **Cite Sources** - When referencing specific information, mention the article source
3. **Be Conversational** - Remember previous messages in our conversation and refer back to them when relevant
4. **Stay Factual** - Base your answers on the provided articles; if information isn't available, say so
5. **Provide Insights** - Help users understand the significance and implications of news
6. **Be Concise but Comprehensive** - Give thorough answers without unnecessary padding

## Formatting Examples:
- Use **bold** for key points
- Use bullet points for lists
- Use ### for section headers when needed
- Use > for notable quotes from articles"""
        else:
            # Prompt for when we couldn't find relevant articles
            system_prompt = """You are an intelligent AI news assistant. The user asked a question, but the available articles don't directly cover their topic of interest.

## Response Guidelines:
1. **Be Honest** - Acknowledge that the available articles don't specifically cover their topic
2. **Summarize What's Available** - If there are any tangentially related articles, briefly mention what they cover
3. **Be Helpful** - Suggest what types of information they might search for, or recommend checking official sources
4. **Stay Professional** - Don't make up information; clearly state what you don't have data on

Start your response by acknowledging the limitation, then provide whatever helpful context you can."""

        # Step 8: Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(history)
        
        # Step 9: Create user prompt with context info
        if is_relevant:
            user_prompt = f"""Based on the following news articles, please answer my question.

**Question:** {message}

**Available Article Context:**
{article_context}

Please provide a helpful, well-formatted response based on the articles above and our conversation history."""
        else:
            # Enhanced prompt when articles aren't relevant
            topics_searched_str = ", ".join(fetched_topics) if fetched_topics else "related topics"
            user_prompt = f"""The user asked: **{message}**

I searched for news about: {topics_searched_str}

**Available Articles (may not directly answer the question):**
{article_context}

Please acknowledge that the available articles don't directly cover "{message}" but provide any related information that might be helpful. Suggest alternative resources if appropriate."""

        messages.append({"role": "user", "content": user_prompt})
        
        # Step 10: Generate response
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            ai_response = f"I apologize, but I encountered an error generating a response. Please try again.\n\nError: {str(e)}"
        
        # Step 11: Store messages in session
        user_msg = ChatMessage(
            role="user",
            content=message,
            timestamp=datetime.utcnow().isoformat()
        )
        
        sources = [
            {
                "article_id": a.get("article_id"),
                "title": a.get("title"),
                "source": a.get("source"),
                "snippet": a.get("content", "")[:150] + "..."
            }
            for a in local_articles
        ]
        
        assistant_msg = ChatMessage(
            role="assistant",
            content=ai_response,
            timestamp=datetime.utcnow().isoformat(),
            sources=sources
        )
        
        self._sessions[session_id].append(user_msg)
        self._sessions[session_id].append(assistant_msg)
        
        # Save session
        self._save_session(session_id)
        
        return ChatResponse(
            response=ai_response,
            sources=sources,
            articles_searched=articles_searched,
            articles_fetched=articles_fetched,
            session_id=session_id,
            fetched_topics=fetched_topics,
            relevance_score=round(relevance_score, 3)
        )


# Singleton instance
_ai_chat_engine: Optional[AIChatEngine] = None


def get_ai_chat_engine() -> AIChatEngine:
    """Get or create the AI Chat engine instance"""
    global _ai_chat_engine
    if _ai_chat_engine is None:
        _ai_chat_engine = AIChatEngine()
    return _ai_chat_engine

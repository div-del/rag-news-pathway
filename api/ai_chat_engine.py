"""
AI Chat Engine - Intelligent chat with news article context.
Uses efficient two-tier retrieval: local search first, then fetch new if needed.
Maintains conversation history for contextual responses.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI

from config import Config
from api.article_store import get_article_store

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
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AIChatEngine:
    """
    AI Chat Engine with intelligent article context retrieval.
    
    Features:
    - Two-tier search: local SQLite first, then Serper API
    - Conversation history management
    - Markdown-formatted responses
    - OpenRouter Mistral integration
    """
    
    def __init__(self):
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
    
    async def _fetch_new_articles(self, query: str, num_results: int = 5) -> int:
        """
        Fetch new articles from Serper API if local search is insufficient.
        Returns the number of new articles fetched.
        """
        try:
            from connectors.news_connector import SerperNewsConnector
            from connectors.article_scraper import ArticleScraper
            
            connector = SerperNewsConnector()
            scraper = ArticleScraper()
            store = get_article_store()
            
            # Search for news
            results = await connector.search_news(query, num_results=num_results)
            
            if not results:
                return 0
            
            count = 0
            for r in results:
                try:
                    # Check if already exists
                    if store.article_exists(r.url):
                        continue
                    
                    # Scrape the article
                    article = await scraper.scrape_article(
                        url=r.url,
                        category=r.category,
                        source=r.source
                    )
                    
                    if article:
                        article_dict = article.to_dict()
                        if store.add_article(article_dict):
                            count += 1
                            
                except Exception as e:
                    logger.debug(f"Error fetching article {r.url}: {e}")
                    continue
            
            logger.info(f"Fetched {count} new articles for query: {query}")
            return count
            
        except Exception as e:
            logger.error(f"Error fetching new articles: {e}")
            return 0
    
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
        
        # Step 1: Search local articles
        local_articles = self._search_local_articles(message, limit=5)
        articles_searched = len(local_articles)
        articles_fetched = 0
        
        # Step 2: If insufficient results, fetch new articles
        if len(local_articles) < 3:
            articles_fetched = await self._fetch_new_articles(message, num_results=5)
            
            if articles_fetched > 0:
                # Re-search to include new articles
                local_articles = self._search_local_articles(message, limit=5)
                articles_searched = len(local_articles)
        
        # Step 3: Build context
        article_context = self._build_context(local_articles)
        
        # Step 4: Build conversation history
        history = self._build_conversation_history(session_id)
        
        # Step 5: Create system prompt
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

        # Step 6: Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(history)
        
        # Add current query with context
        user_prompt = f"""Based on the following news articles, please answer my question.

**Question:** {message}

**Available Article Context:**
{article_context}

Please provide a helpful, well-formatted response based on the articles above and our conversation history."""

        messages.append({"role": "user", "content": user_prompt})
        
        # Step 7: Generate response
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
        
        # Step 8: Store messages in session
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
            session_id=session_id
        )


# Singleton instance
_ai_chat_engine: Optional[AIChatEngine] = None


def get_ai_chat_engine() -> AIChatEngine:
    """Get or create the AI Chat engine instance"""
    global _ai_chat_engine
    if _ai_chat_engine is None:
        _ai_chat_engine = AIChatEngine()
    return _ai_chat_engine

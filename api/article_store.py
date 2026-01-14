"""
Article Store - SQLite CRUD operations for articles.
Provides persistent storage for articles across server restarts.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from api.db_models import Base, Article

logger = logging.getLogger(__name__)

# Database path (same directory as onboarding.db)
ARTICLES_DB_PATH = Path(__file__).parent.parent / "articles.db"


class ArticleStore:
    """
    SQLite-based article storage.
    Handles persistence of articles across server restarts.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._engine = create_engine(f"sqlite:///{ARTICLES_DB_PATH}", echo=False)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        self._initialized = True
        
        logger.info(f"ArticleStore initialized with database: {ARTICLES_DB_PATH}")
    
    def _get_session(self) -> Session:
        return self._Session()
    
    def add_article(self, article_data: Dict[str, Any]) -> bool:
        """
        Add a new article to the store.
        Returns True if added, False if already exists.
        """
        session = self._get_session()
        try:
            # Check if article already exists
            existing = session.query(Article).filter_by(
                article_id=article_data.get("article_id")
            ).first()
            
            if existing:
                logger.debug(f"Article already exists: {article_data.get('article_id')}")
                session.close()
                return False
            
            # Create new article
            article = Article(
                article_id=article_data.get("article_id"),
                url=article_data.get("url"),
                title=article_data.get("title"),
                content=article_data.get("content"),
                author=article_data.get("author"),
                category=article_data.get("category"),
                source=article_data.get("source"),
                published_date=article_data.get("publish_date"),
                description=article_data.get("description"),
                image_url=article_data.get("image_url"),
                language=article_data.get("language"),
                topics=article_data.get("topics", []),
            )
            
            session.add(article)
            session.commit()
            logger.info(f"Added article: {article_data.get('title', '')[:50]}...")
            session.close()
            return True
            
        except IntegrityError:
            session.rollback()
            session.close()
            logger.debug(f"Article already exists (integrity): {article_data.get('url')}")
            return False
        except Exception as e:
            session.rollback()
            session.close()
            logger.error(f"Error adding article: {e}")
            return False
    
    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get a single article by ID"""
        session = self._get_session()
        try:
            article = session.query(Article).filter_by(article_id=article_id).first()
            if article:
                result = self._article_to_dict(article)
                session.close()
                return result
            session.close()
            return None
        except Exception as e:
            session.close()
            logger.error(f"Error getting article: {e}")
            return None
    
    def get_all_articles(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all articles from the store"""
        session = self._get_session()
        try:
            articles = session.query(Article).order_by(
                Article.scraped_at.desc()
            ).limit(limit).all()
            
            result = [self._article_to_dict(a) for a in articles]
            session.close()
            return result
        except Exception as e:
            session.close()
            logger.error(f"Error getting all articles: {e}")
            return []
    
    def get_articles_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles filtered by category"""
        session = self._get_session()
        try:
            articles = session.query(Article).filter(
                Article.category.ilike(f"%{category}%")
            ).order_by(Article.scraped_at.desc()).limit(limit).all()
            
            result = [self._article_to_dict(a) for a in articles]
            session.close()
            return result
        except Exception as e:
            session.close()
            logger.error(f"Error getting articles by category: {e}")
            return []
    
    def search_articles(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Simple text search in title and content"""
        session = self._get_session()
        try:
            articles = session.query(Article).filter(
                (Article.title.ilike(f"%{query}%")) |
                (Article.content.ilike(f"%{query}%"))
            ).order_by(Article.scraped_at.desc()).limit(limit).all()
            
            result = [self._article_to_dict(a) for a in articles]
            session.close()
            return result
        except Exception as e:
            session.close()
            logger.error(f"Error searching articles: {e}")
            return []
    
    def get_article_count(self) -> int:
        """Get total number of articles"""
        session = self._get_session()
        try:
            count = session.query(Article).count()
            session.close()
            return count
        except Exception as e:
            session.close()
            logger.error(f"Error getting article count: {e}")
            return 0
    
    def article_exists(self, url: str) -> bool:
        """Check if article with URL already exists"""
        session = self._get_session()
        try:
            exists = session.query(Article).filter_by(url=url).first() is not None
            session.close()
            return exists
        except Exception as e:
            session.close()
            return False
    
    def _article_to_dict(self, article: Article) -> Dict[str, Any]:
        """Convert Article model to dictionary"""
        return {
            "article_id": article.article_id,
            "url": article.url,
            "title": article.title,
            "content": article.content,
            "author": article.author,
            "category": article.category,
            "source": article.source,
            "publish_date": article.published_date,
            "description": article.description,
            "image_url": article.image_url,
            "language": article.language,
            "topics": article.topics or [],
            "scraped_at": article.scraped_at.isoformat() if article.scraped_at else None,
        }


# Singleton accessor
_article_store: Optional[ArticleStore] = None


def get_article_store() -> ArticleStore:
    """Get or create the article store instance"""
    global _article_store
    if _article_store is None:
        _article_store = ArticleStore()
    return _article_store

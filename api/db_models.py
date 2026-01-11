"""
Database models for the Live AI News Platform.
Defines tables for users, preferences, articles, interactions, and comparisons.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, 
    ForeignKey, Boolean, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """User account table"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    interactions = relationship("UserInteraction", back_populates="user", cascade="all, delete-orphan")


class UserPreference(Base):
    """User preference scores for categories and topics"""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    
    # Preference type: 'category' or 'topic'
    preference_type = Column(String(50), nullable=False)
    preference_value = Column(String(255), nullable=False)  # e.g., 'Technology', 'Tesla'
    
    # Affinity score (0.0 to 1.0)
    score = Column(Float, default=0.5)
    
    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    __table_args__ = (
        Index('idx_user_pref_type_value', 'user_id', 'preference_type', 'preference_value'),
    )


class Article(Base):
    """News article metadata"""
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Article content
    url = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text)
    author = Column(String(255))
    
    # Metadata
    category = Column(String(100), index=True)
    source = Column(String(255))
    published_date = Column(DateTime, index=True)
    
    # Topics extracted (JSON array)
    topics = Column(JSON)  # e.g., ['Tesla', 'EV', 'Technology']
    
    # Timestamps
    ingested_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    interactions = relationship("UserInteraction", back_populates="article", cascade="all, delete-orphan")


class UserInteraction(Base):
    """Track user interactions with articles"""
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    article_id = Column(String(255), ForeignKey("articles.article_id", ondelete="CASCADE"), nullable=False)
    
    # Interaction type: 'view', 'chat', 'like', 'share', 'compare'
    interaction_type = Column(String(50), nullable=False, index=True)
    
    # Additional data (e.g., chat messages, time spent)
    metadata = Column(JSON)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="interactions")
    article = relationship("Article", back_populates="interactions")
    
    __table_args__ = (
        Index('idx_user_article', 'user_id', 'article_id'),
        Index('idx_user_interaction_type', 'user_id', 'interaction_type'),
    )


class ArticleComparison(Base):
    """Store article comparison sessions"""
    __tablename__ = "article_comparisons"
    
    id = Column(Integer, primary_key=True, index=True)
    comparison_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    
    # Articles being compared (JSON array of article_ids)
    article_ids = Column(JSON, nullable=False)
    
    # Comparison query
    query = Column(Text)
    
    # AI response
    response = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

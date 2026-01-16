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


class ClerkUser(Base):
    """Clerk authenticated user"""
    __tablename__ = "clerk_users"
    
    id = Column(Integer, primary_key=True, index=True)
    clerk_id = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    image_url = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
    url = Column(Text, nullable=False, unique=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    author = Column(String(255))
    
    # Metadata
    category = Column(String(100), index=True)
    source = Column(String(255))
    published_date = Column(String(255))  # Store as string for compatibility
    description = Column(Text, nullable=True)
    image_url = Column(Text, nullable=True)
    language = Column(String(10), nullable=True)
    
    # Topics extracted (JSON array)
    topics = Column(JSON)  # e.g., ['Tesla', 'EV', 'Technology']
    
    # Timestamps
    scraped_at = Column(DateTime, default=datetime.utcnow, index=True)
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
    extra_data = Column(JSON)
    
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


class UserOnboarding(Base):
    """Store user onboarding questionnaire responses (one-time)"""
    __tablename__ = "user_onboarding"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Q1: News categories of interest (multi-select)
    categories = Column(JSON)  # e.g., ['Technology', 'Business', 'Sports']
    
    # Q2: Reading depth preference
    reading_depth = Column(String(50))  # 'headlines', 'quick', 'in_depth', 'mixed'
    
    # Q5: Daily time spent on news
    daily_time = Column(String(50))  # 'less_10', '10_30', '30_60', 'more_60'
    
    # Q6: Preferred content formats (multi-select)
    content_formats = Column(JSON)  # e.g., ['text', 'video', 'audio', 'infographics']
    
    # Q7: Primary reason for staying informed
    primary_reason = Column(String(100))  # 'professional', 'personal', 'investment', 'social', 'academic'
    
    # Q8: Work industry
    industry = Column(String(100))  # e.g., 'Technology', 'Finance', 'Healthcare'
    
    # Q9: Regional preferences (multi-select)
    regions = Column(JSON)  # e.g., ['Local', 'National', 'International']
    
    # Q12: AI summary preference
    ai_summary_preference = Column(String(50))  # 'love_it', 'useful_verify', 'prefer_human', 'open_to_try'
    
    # Q15: Importance ratings (1-5 scale)
    importance_timely = Column(Integer)  # How important is breaking news
    importance_accurate = Column(Integer)  # How important is fact-checked content
    importance_engaging = Column(Integer)  # How important is well-written content
    
    # Metadata
    completed_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatSession(Base):
    """Store AI chat sessions with conversation history"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    
    # Conversation messages (JSON array of {role, content, timestamp, sources})
    messages = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


"""
User Profile Management.
Handles user preferences, interaction tracking, and preference scoring.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from collections import defaultdict

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class UserPreferenceScore:
    """Represents a user's preference score for a topic/category"""
    preference_type: str  # "category" or "topic"
    preference_value: str
    score: float
    interaction_count: int
    last_interaction: str


@dataclass
class UserProfile:
    """Complete user profile"""
    user_id: str
    created_at: str
    last_active: str
    
    # Preferences
    category_preferences: Dict[str, float] = field(default_factory=dict)
    topic_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Interaction history
    viewed_articles: List[str] = field(default_factory=list)
    chatted_articles: List[str] = field(default_factory=list)
    compared_articles: List[List[str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_top_categories(self, n: int = 5) -> List[str]:
        """Get user's top N preferred categories"""
        sorted_cats = sorted(
            self.category_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [cat for cat, _ in sorted_cats[:n]]
    
    def get_top_topics(self, n: int = 10) -> List[str]:
        """Get user's top N preferred topics"""
        sorted_topics = sorted(
            self.topic_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [topic for topic, _ in sorted_topics[:n]]


class UserProfileManager:
    """
    Manages user profiles and tracks interactions.
    Updates preferences based on user behavior.
    """
    
    def __init__(
        self,
        score_increment: float = None,
        decay_factor: float = None
    ):
        self.score_increment = score_increment or Config.PREFERENCE_SCORE_INCREMENT
        self.decay_factor = decay_factor or Config.PREFERENCE_DECAY_FACTOR
        
        # In-memory user storage (will be backed by database)
        self._users: Dict[str, UserProfile] = {}
        
        # Interaction weights
        self._weights = {
            "view": 1.0,
            "chat": 2.0,
            "compare": 1.5,
            "like": 3.0,
            "share": 2.5
        }
        
        logger.info("User profile manager initialized")
    
    def get_or_create_user(self, user_id: str) -> UserProfile:
        """Get existing user or create new one"""
        if user_id not in self._users:
            now = datetime.utcnow().isoformat()
            self._users[user_id] = UserProfile(
                user_id=user_id,
                created_at=now,
                last_active=now
            )
            logger.info(f"Created new user profile: {user_id}")
        
        return self._users[user_id]
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile if exists"""
        return self._users.get(user_id)
    
    def _update_preference(
        self,
        preferences: Dict[str, float],
        key: str,
        weight: float = 1.0
    ) -> float:
        """Update a preference score with decay"""
        current = preferences.get(key, 0.5)
        
        # Increment with weight
        increment = self.score_increment * weight
        new_score = min(1.0, current + increment)
        
        preferences[key] = new_score
        return new_score
    
    def track_interaction(
        self,
        user_id: str,
        article: Dict[str, Any],
        interaction_type: str
    ) -> Dict[str, Any]:
        """
        Track a user interaction and update preferences.
        
        Args:
            user_id: User identifier
            article: Article data with category, topics, etc.
            interaction_type: Type of interaction (view, chat, compare, like, share)
        
        Returns:
            Updated preference changes
        """
        user = self.get_or_create_user(user_id)
        user.last_active = datetime.utcnow().isoformat()
        
        weight = self._weights.get(interaction_type, 1.0)
        article_id = article.get("article_id", "")
        category = article.get("category", "")
        topics = article.get("topics", [])
        
        if isinstance(topics, str):
            topics = json.loads(topics) if topics else []
        
        changes = {
            "user_id": user_id,
            "article_id": article_id,
            "interaction_type": interaction_type,
            "preference_updates": []
        }
        
        # Update category preference
        if category:
            new_score = self._update_preference(
                user.category_preferences,
                category,
                weight
            )
            changes["preference_updates"].append({
                "type": "category",
                "value": category,
                "new_score": new_score
            })
        
        # Update topic preferences
        for topic in topics:
            new_score = self._update_preference(
                user.topic_preferences,
                topic,
                weight * 0.5  # Topics get half the weight of categories
            )
            changes["preference_updates"].append({
                "type": "topic",
                "value": topic,
                "new_score": new_score
            })
        
        # Track in interaction history
        if interaction_type == "view" and article_id not in user.viewed_articles:
            user.viewed_articles.append(article_id)
        elif interaction_type == "chat" and article_id not in user.chatted_articles:
            user.chatted_articles.append(article_id)
        
        logger.debug(f"Tracked {interaction_type} for user {user_id} on article {article_id}")
        return changes
    
    def track_comparison(
        self,
        user_id: str,
        articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Track a comparison between multiple articles"""
        user = self.get_or_create_user(user_id)
        
        article_ids = [a.get("article_id", "") for a in articles]
        user.compared_articles.append(article_ids)
        
        changes = {"user_id": user_id, "compared_articles": article_ids, "preference_updates": []}
        
        # Update preferences for all articles in comparison
        for article in articles:
            update = self.track_interaction(user_id, article, "compare")
            changes["preference_updates"].extend(update.get("preference_updates", []))
        
        return changes
    
    def apply_decay(self, user_id: str):
        """Apply decay to user preferences (call periodically)"""
        user = self.get_user(user_id)
        if not user:
            return
        
        # Decay category preferences
        for key in user.category_preferences:
            user.category_preferences[key] *= self.decay_factor
            if user.category_preferences[key] < 0.1:
                user.category_preferences[key] = 0.1
        
        # Decay topic preferences
        for key in user.topic_preferences:
            user.topic_preferences[key] *= self.decay_factor
            if user.topic_preferences[key] < 0.1:
                user.topic_preferences[key] = 0.1
        
        logger.debug(f"Applied preference decay for user {user_id}")
    
    def get_user_preferences_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user preferences"""
        user = self.get_user(user_id)
        if not user:
            return {"error": "User not found"}
        
        return {
            "user_id": user_id,
            "top_categories": user.get_top_categories(5),
            "top_topics": user.get_top_topics(10),
            "total_views": len(user.viewed_articles),
            "total_chats": len(user.chatted_articles),
            "total_comparisons": len(user.compared_articles),
            "last_active": user.last_active
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            "total_users": len(self._users),
            "score_increment": self.score_increment,
            "decay_factor": self.decay_factor
        }


# Singleton instance
_profile_manager: Optional[UserProfileManager] = None


def get_profile_manager() -> UserProfileManager:
    """Get or create the profile manager instance"""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = UserProfileManager()
    return _profile_manager


async def test_profile_manager():
    """Test the profile manager"""
    manager = UserProfileManager()
    
    # Create test user
    user = manager.get_or_create_user("test_user_123")
    print(f"Created user: {user.user_id}")
    
    # Simulate interactions
    test_article = {
        "article_id": "tesla_001",
        "title": "Tesla News",
        "category": "Technology",
        "topics": ["Tesla", "EV", "automotive"]
    }
    
    # Track view
    changes = manager.track_interaction("test_user_123", test_article, "view")
    print(f"\nAfter view: {changes}")
    
    # Track chat
    changes = manager.track_interaction("test_user_123", test_article, "chat")
    print(f"\nAfter chat: {changes}")
    
    # Get summary
    summary = manager.get_user_preferences_summary("test_user_123")
    print(f"\nUser summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_profile_manager())

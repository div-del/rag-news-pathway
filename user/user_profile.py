"""
User Profile Management.
Handles user preferences, interaction tracking, and preference scoring.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

from config import Config
from api.database import get_db_session
from api.db_models import User, UserPreference, UserInteraction, UserOnboarding

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Complete user profile"""
    user_id: str
    created_at: str
    last_active: str
    
    # Preferences
    category_preferences: Dict[str, float] = field(default_factory=dict)
    topic_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Interaction history (ids only for lightweight access)
    viewed_articles: List[str] = field(default_factory=list)
    chatted_articles: List[str] = field(default_factory=list)
    
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
    Persists data to PostgreSQL.
    """
    
    def __init__(
        self,
        score_increment: float = None,
        decay_factor: float = None
    ):
        self.score_increment = score_increment or Config.PREFERENCE_SCORE_INCREMENT
        self.decay_factor = decay_factor or Config.PREFERENCE_DECAY_FACTOR
        
        # Interaction weights
        self._weights = {
            "view": 1.0,
            "filter": 1.5,  # [NEW] Explicit filtering is a strong signal
            "chat": 2.0,
            "compare": 1.5,
            "like": 3.0,
            "share": 2.5
        }
        
        logger.info("User profile manager initialized (DB-backed)")
    
    def get_or_create_user(self, user_id: str) -> UserProfile:
        """Get existing user or create new one (with DB load)"""
        if not user_id:
            return None

        with get_db_session() as session:
            # Try to get user from DB
            db_user = session.query(User).filter_by(user_id=user_id).first()
            
            if not db_user:
                # Create new user
                now = datetime.utcnow()
                db_user = User(user_id=user_id, created_at=now, last_active=now)
                session.add(db_user)
                session.commit()
                logger.info(f"Created new user profile: {user_id}")
            
            # Load preferences
            prefs = session.query(UserPreference).filter_by(user_id=user_id).all()
            category_prefs = {p.preference_value: p.score for p in prefs if p.preference_type == 'category'}
            topic_prefs = {p.preference_value: p.score for p in prefs if p.preference_type == 'topic'}
            
            # Load interactions (just IDs for cache)
            # Optimization: could limit to recent ones if list gets too long
            interactions = session.query(UserInteraction).filter_by(user_id=user_id).all()
            viewed = [i.article_id for i in interactions if i.interaction_type == 'view']
            chatted = [i.article_id for i in interactions if i.interaction_type == 'chat']

            return UserProfile(
                user_id=user_id,
                created_at=db_user.created_at.isoformat(),
                last_active=db_user.last_active.isoformat(),
                category_preferences=category_prefs,
                topic_preferences=topic_prefs,
                viewed_articles=viewed,
                chatted_articles=chatted
            )
            
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile if exists"""
        return self.get_or_create_user(user_id)
    
    def _update_preference_db(
        self,
        session,
        user_id: str,
        pref_type: str,
        value: str,
        weight: float
    ) -> float:
        """Update a preference score in DB"""
        pref = session.query(UserPreference).filter_by(
            user_id=user_id, preference_type=pref_type, preference_value=value
        ).first()
        
        if not pref:
            # Start at base 0.5
            current_score = 0.5
            pref = UserPreference(
                user_id=user_id,
                preference_type=pref_type,
                preference_value=value,
                score=current_score
            )
            session.add(pref)
        else:
            current_score = pref.score
            
        increment = self.score_increment * weight
        new_score = min(1.0, current_score + increment)
        
        pref.score = new_score
        pref.updated_at = datetime.utcnow()
        
        return new_score

    def track_interaction(
        self,
        user_id: str,
        article: Dict[str, Any],
        interaction_type: str
    ) -> Dict[str, Any]:
        """
        Track interaction and persist to DB.
        """
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
        
        with get_db_session() as session:
            # 1. Log interaction
            interaction = UserInteraction(
                user_id=user_id,
                article_id=article_id,
                interaction_type=interaction_type,
                created_at=datetime.utcnow()
            )
            session.add(interaction)
            
            # 2. Update User last_active
            db_user = session.query(User).filter_by(user_id=user_id).first()
            if db_user:
                db_user.last_active = datetime.utcnow()
            
            # 3. Update Category Preference
            if category:
                new_score = self._update_preference_db(session, user_id, "category", category, weight)
                changes["preference_updates"].append({
                    "type": "category", "value": category, "new_score": new_score
                })
                
            # 4. Update Topic Preferences
            for topic in topics:
                new_score = self._update_preference_db(session, user_id, "topic", topic, weight * 0.5)
                changes["preference_updates"].append({
                    "type": "topic", "value": topic, "new_score": new_score
                })
                
            session.commit()
            
        return changes

    def track_filter(self, user_id: str, category: str) -> bool:
        """
        [NEW] Track robustly when a user actively filters by a category.
        This provides a strong signal of interest.
        """
        if not category:
            return False
            
        with get_db_session() as session:
            # Update User last_active
            db_user = session.query(User).filter_by(user_id=user_id).first()
            if db_user:
                db_user.last_active = datetime.utcnow()
                
            # Boost category score significantly (weight 1.5)
            self._update_preference_db(session, user_id, "category", category, self._weights["filter"])
            session.commit()
            
        logger.info(f"Tracked filter '{category}' for user {user_id}")
        return True

    def seed_from_onboarding(self, user_id: str, categories: List[str]):
        """
        [NEW] Initialize user preferences from Onboarding data.
        """
        if not categories:
            return

        with get_db_session() as session:
            # Update User last_active
            db_user = session.query(User).filter_by(user_id=user_id).first()
            if db_user:
                db_user.last_active = datetime.utcnow()
            
            for cat in categories:
                # Give a strong initial boost (0.8) for explicit onboarding selection
                pref = session.query(UserPreference).filter_by(
                    user_id=user_id, preference_type="category", preference_value=cat
                ).first()
                
                if not pref:
                    pref = UserPreference(
                        user_id=user_id,
                        preference_type="category",
                        preference_value=cat,
                        score=0.8  # High initial score
                    )
                    session.add(pref)
                else:
                    pref.score = max(pref.score, 0.8)
            
            session.commit()
        
        logger.info(f"Seeded preferences for {user_id} from onboarding categories: {categories}")

    def track_comparison(self, user_id: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track comparison in DB"""
        # Comparison logic remains similar, just iterates through track_interaction
        changes = {"user_id": user_id, "preference_updates": []}
        
        for article in articles:
            update = self.track_interaction(user_id, article, "compare")
            changes["preference_updates"].extend(update.get("preference_updates", []))
            
        # In a real expanded version, we'd log to ArticleComparison table too
        # But UserInteraction is sufficient for preference learning
        
        return changes
    
    def get_user_preferences_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary from DB"""
        user = self.get_or_create_user(user_id)
        if not user:
            return {"error": "User not found"}
            
        return {
            "user_id": user_id,
            "top_categories": user.get_top_categories(5),
            "top_topics": user.get_top_topics(10),
            "total_views": len(user.viewed_articles),
            "total_chats": len(user.chatted_articles),
            "last_active": user.last_active
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats"""
        return {
            "mode": "persistent (postgres)",
            "score_increment": self.score_increment
        }


# Singleton instance
_profile_manager: Optional[UserProfileManager] = None


def get_profile_manager() -> UserProfileManager:
    """Get or create the profile manager instance"""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = UserProfileManager()
    return _profile_manager


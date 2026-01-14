"""
Recommendation Engine.
Provides personalized article recommendations based on user preferences.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

from config import Config
from user.user_profile import UserProfileManager, get_profile_manager

logger = logging.getLogger(__name__)


@dataclass
class ArticleRecommendation:
    """Represents an article recommendation"""
    article_id: str
    title: str
    score: float
    reasons: List[str]
    category: str
    topics: List[str]


@dataclass
class ComparisonSuggestion:
    """Suggestion for article comparison"""
    article_ids: List[str]
    titles: List[str]
    common_topics: List[str]
    comparison_prompt: str


class RecommendationEngine:
    """
    Provides personalized recommendations based on user preferences.
    Features:
    - Personalized article ranking
    - Smart comparison suggestions (Tesla vs BMW style)
    - Topic-based clustering
    """
    
    def __init__(self, profile_manager: UserProfileManager = None):
        self.profile_manager = profile_manager or get_profile_manager()
        
        # In-memory article store (shared with RAG engine)
        self._articles: Dict[str, Dict[str, Any]] = {}
        
        # Comparison topic pairs (for smart suggestions)
        self._comparison_pairs = [
            (["Tesla", "EV"], ["BMW", "Mercedes", "Audi"]),
            (["Apple", "iPhone"], ["Samsung", "Google", "Android"]),
            (["Microsoft", "Windows"], ["Apple", "Mac"]),
            (["Amazon", "AWS"], ["Google Cloud", "Azure"]),
            (["Facebook", "Meta"], ["Twitter", "X", "TikTok"]),
        ]
        
        logger.info("Recommendation engine initialized")
    
    def add_article(self, article: Dict[str, Any]):
        """Add an article to the recommendation pool"""
        article_id = article.get("article_id")
        if article_id:
            self._articles[article_id] = article
    
    def _score_article_for_user(
        self,
        article: Dict[str, Any],
        user_id: str
    ) -> Tuple[float, List[str]]:
        """
        Score an article for a specific user based on preferences.
        
        Returns:
            Tuple of (score, reasons)
        """
        user = self.profile_manager.get_user(user_id)
        if not user:
            return 0.5, ["No user profile"]
        
        score = 0.5  # Base score
        reasons = []
        
        category = article.get("category", "")
        topics = article.get("topics", [])
        if isinstance(topics, str):
            topics = json.loads(topics) if topics else []
        
        # Category preference
        cat_pref = user.category_preferences.get(category, 0.5)
        if cat_pref > 0.6:
            score += (cat_pref - 0.5) * 0.5
            reasons.append(f"Interested in {category}")
        
        # Topic preferences
        topic_boost = 0
        matched_topics = []
        for topic in topics:
            topic_pref = user.topic_preferences.get(topic, 0)
            if topic_pref > 0.5:
                topic_boost += topic_pref * 0.1
                matched_topics.append(topic)
        
        if matched_topics:
            score += min(topic_boost, 0.3)  # Cap topic boost
            reasons.append(f"Matches interests: {', '.join(matched_topics[:3])}")
        
        # Recency bonus
        publish_date = article.get("publish_date")
        if publish_date:
            try:
                pub_dt = datetime.fromisoformat(publish_date.replace("Z", "+00:00"))
                hours_old = (datetime.utcnow() - pub_dt.replace(tzinfo=None)).total_seconds() / 3600
                if hours_old < 24:
                    score += 0.1
                    reasons.append("Fresh news")
            except:
                pass
        
        # Novelty penalty (already viewed)
        article_id = article.get("article_id", "")
        if article_id in user.viewed_articles:
            score -= 0.2
            reasons.append("Already viewed")
        
        return min(1.0, max(0.0, score)), reasons
    
    def get_personalized_feed(
        self,
        user_id: str,
        limit: int = 20,
        exclude_viewed: bool = True
    ) -> List[ArticleRecommendation]:
        """
        Get personalized article feed for a user.
        
        Args:
            user_id: User identifier
            limit: Max articles to return
            exclude_viewed: Whether to exclude already viewed articles
        
        Returns:
            List of ArticleRecommendation sorted by score
        """
        user = self.profile_manager.get_or_create_user(user_id)
        
        recommendations = []
        
        for article_id, article in self._articles.items():
            # Skip viewed if requested
            if exclude_viewed and article_id in user.viewed_articles:
                continue
            
            score, reasons = self._score_article_for_user(article, user_id)
            
            topics = article.get("topics", [])
            if isinstance(topics, str):
                topics = json.loads(topics) if topics else []
            
            rec = ArticleRecommendation(
                article_id=article_id,
                title=article.get("title", "Untitled"),
                score=score,
                reasons=reasons,
                category=article.get("category", ""),
                topics=topics
            )
            recommendations.append(rec)
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return recommendations[:limit]
    
    def get_related_articles(
        self,
        article_id: str,
        limit: int = 5
    ) -> List[ArticleRecommendation]:
        """
        Get articles related to a specific article.
        Used for "You might also like" suggestions.
        """
        source_article = self._articles.get(article_id)
        if not source_article:
            return []
        
        source_topics = source_article.get("topics", [])
        if isinstance(source_topics, str):
            source_topics = json.loads(source_topics) if source_topics else []
        source_topics_set = set(t.lower() for t in source_topics)
        
        source_category = source_article.get("category", "").lower()
        
        scored_articles = []
        
        for aid, article in self._articles.items():
            if aid == article_id:
                continue
            
            topics = article.get("topics", [])
            if isinstance(topics, str):
                topics = json.loads(topics) if topics else []
            topics_set = set(t.lower() for t in topics)
            
            # Calculate similarity score
            topic_overlap = len(source_topics_set & topics_set)
            category_match = 1 if article.get("category", "").lower() == source_category else 0
            
            score = topic_overlap * 0.3 + category_match * 0.2
            
            if score > 0:
                reasons = []
                if topic_overlap:
                    common = source_topics_set & topics_set
                    reasons.append(f"Common topics: {', '.join(list(common)[:3])}")
                if category_match:
                    reasons.append(f"Same category: {source_category}")
                
                rec = ArticleRecommendation(
                    article_id=aid,
                    title=article.get("title", "Untitled"),
                    score=score,
                    reasons=reasons,
                    category=article.get("category", ""),
                    topics=topics
                )
                scored_articles.append(rec)
        
        scored_articles.sort(key=lambda x: x.score, reverse=True)
        return scored_articles[:limit]
    
    def suggest_comparison(
        self,
        article_id: str
    ) -> Optional[ComparisonSuggestion]:
        """
        Suggest a comparison for a given article.
        E.g., if user is reading Tesla news, suggest comparing with BMW.
        """
        source_article = self._articles.get(article_id)
        if not source_article:
            return None
        
        source_topics = source_article.get("topics", [])
        if isinstance(source_topics, str):
            source_topics = json.loads(source_topics) if source_topics else []
        source_topics_lower = [t.lower() for t in source_topics]
        
        # Find matching comparison pair
        target_topics = []
        for group_a, group_b in self._comparison_pairs:
            group_a_lower = [t.lower() for t in group_a]
            group_b_lower = [t.lower() for t in group_b]
            
            if any(t in group_a_lower for t in source_topics_lower):
                target_topics = group_b
                break
            elif any(t in group_b_lower for t in source_topics_lower):
                target_topics = group_a
                break
        
        if not target_topics:
            return None
        
        # Find articles matching target topics
        for aid, article in self._articles.items():
            if aid == article_id:
                continue
            
            topics = article.get("topics", [])
            if isinstance(topics, str):
                topics = json.loads(topics) if topics else []
            
            if any(t.lower() in [tt.lower() for tt in topics] for t in target_topics):
                # Found a match!
                common = set(source_topics) & set(topics)
                
                return ComparisonSuggestion(
                    article_ids=[article_id, aid],
                    titles=[source_article.get("title", ""), article.get("title", "")],
                    common_topics=list(common),
                    comparison_prompt=f"Compare these two articles about related topics"
                )
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "total_articles": len(self._articles),
            "comparison_pairs": len(self._comparison_pairs)
        }


# Singleton instance
_recommendation_engine: Optional[RecommendationEngine] = None


def get_recommendation_engine() -> RecommendationEngine:
    """Get or create the recommendation engine instance"""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine


async def test_recommendation_engine():
    """Test the recommendation engine"""
    engine = RecommendationEngine()
    
    # Add test articles
    engine.add_article({
        "article_id": "tesla_001",
        "title": "Tesla Announces New Model Y",
        "category": "Technology",
        "topics": ["Tesla", "EV", "automotive"]
    })
    
    engine.add_article({
        "article_id": "bmw_001",
        "title": "BMW Launches Electric iX Series",
        "category": "Technology",
        "topics": ["BMW", "EV", "automotive", "luxury"]
    })
    
    engine.add_article({
        "article_id": "apple_001",
        "title": "Apple Unveils New iPhone",
        "category": "Technology",
        "topics": ["Apple", "iPhone", "smartphone"]
    })
    
    # Test personalized feed
    print("\n=== Personalized Feed ===")
    feed = engine.get_personalized_feed("test_user")
    for rec in feed:
        print(f"  {rec.title} (score: {rec.score:.2f})")
    
    # Test related articles
    print("\n=== Related to Tesla Article ===")
    related = engine.get_related_articles("tesla_001")
    for rec in related:
        print(f"  {rec.title} - {rec.reasons}")
    
    # Test comparison suggestion
    print("\n=== Comparison Suggestion ===")
    suggestion = engine.suggest_comparison("tesla_001")
    if suggestion:
        print(f"  Compare: {suggestion.titles}")
        print(f"  Prompt: {suggestion.comparison_prompt}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import asyncio
    asyncio.run(test_recommendation_engine())

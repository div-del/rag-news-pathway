"""
Web Push Notifications module.
Handles VAPID keys, subscriptions, and sending push notifications.
"""

import os
import json
import logging
from typing import Optional, List
from pathlib import Path
from pywebpush import webpush, WebPushException

logger = logging.getLogger(__name__)

# VAPID configuration
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY_FILE = os.getenv("VAPID_PRIVATE_KEY_FILE", "private_key.pem")
VAPID_EMAIL = os.getenv("VAPID_EMAIL", "admin@livelens.app")

# In-memory subscription store (for demo - use DB in production)
_subscriptions: dict = {}


def get_vapid_public_key() -> str:
    """Get the VAPID public key for frontend subscription"""
    return VAPID_PUBLIC_KEY


def save_subscription(user_id: str, subscription: dict) -> bool:
    """Save a push subscription for a user"""
    try:
        _subscriptions[user_id] = subscription
        logger.info(f"Saved push subscription for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving subscription: {e}")
        return False


def remove_subscription(user_id: str) -> bool:
    """Remove a push subscription for a user"""
    try:
        if user_id in _subscriptions:
            del _subscriptions[user_id]
            logger.info(f"Removed push subscription for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error removing subscription: {e}")
        return False


def get_subscription(user_id: str) -> Optional[dict]:
    """Get subscription for a specific user"""
    return _subscriptions.get(user_id)


def get_all_subscriptions() -> List[dict]:
    """Get all subscriptions for broadcasting"""
    return list(_subscriptions.values())


def send_push_notification(
    subscription: dict,
    title: str,
    body: str,
    url: str = "/app",
    article_id: Optional[str] = None,
    breaking: bool = False
) -> bool:
    """Send a push notification to a single subscription"""
    
    payload = json.dumps({
        "title": title,
        "body": body,
        "url": url,
        "articleId": article_id,
        "breaking": breaking,
        "tag": f"article-{article_id}" if article_id else "livelens-news"
    })
    
    # Get private key path
    private_key_path = Path(VAPID_PRIVATE_KEY_FILE)
    if not private_key_path.is_absolute():
        private_key_path = Path(__file__).parent.parent / VAPID_PRIVATE_KEY_FILE
    
    vapid_claims = {
        "sub": f"mailto:{VAPID_EMAIL}"
    }
    
    try:
        webpush(
            subscription_info=subscription,
            data=payload,
            vapid_private_key=str(private_key_path),
            vapid_claims=vapid_claims
        )
        logger.info(f"Push notification sent successfully")
        return True
    except WebPushException as e:
        logger.error(f"WebPush error: {e}")
        # If subscription is invalid, it should be removed
        if e.response and e.response.status_code in [404, 410]:
            logger.info("Subscription expired or invalid")
        return False
    except Exception as e:
        logger.error(f"Error sending push: {e}")
        return False


def broadcast_notification(
    title: str,
    body: str,
    url: str = "/app",
    article_id: Optional[str] = None,
    breaking: bool = False
) -> dict:
    """Send notification to all subscribers"""
    
    subscriptions = get_all_subscriptions()
    success_count = 0
    failed_count = 0
    
    for sub in subscriptions:
        if send_push_notification(sub, title, body, url, article_id, breaking):
            success_count += 1
        else:
            failed_count += 1
    
    return {
        "total": len(subscriptions),
        "success": success_count,
        "failed": failed_count
    }


def send_breaking_news_alert(article: dict) -> dict:
    """Send a breaking news notification for a new article"""
    
    title = "🔴 Breaking News"
    body = article.get("title", "New article available")[:100]
    url = f"/app?article={article.get('article_id', '')}"
    
    return broadcast_notification(
        title=title,
        body=body,
        url=url,
        article_id=article.get("article_id"),
        breaking=True
    )

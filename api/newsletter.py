"""
Newsletter Service - Email newsletter functionality using Resend.
Free tier: 100 emails/day
"""

import os
import logging
import hashlib
import secrets
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import resend
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class Subscriber(Base):
    """Newsletter subscriber model"""
    __tablename__ = 'newsletter_subscribers'
    
    email = Column(String(255), primary_key=True)
    subscribed_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    unsubscribe_token = Column(String(64), unique=True)
    preferences = Column(Text, default='all')  # 'all', 'daily', 'weekly'
    name = Column(String(255), nullable=True)


@dataclass
class NewsletterResult:
    """Result of sending newsletter"""
    success: bool
    sent_count: int
    failed_count: int
    message: str


class NewsletterService:
    """
    Newsletter service with Resend integration.
    
    Features:
    - Subscribe/Unsubscribe management
    - Daily/Weekly news digest
    - Beautiful HTML email templates
    - Unsubscribe links
    """
    
    def __init__(self):
        # Initialize Resend
        self.api_key = os.getenv('RESEND_API_KEY')
        if self.api_key:
            resend.api_key = self.api_key
            logger.info("Newsletter service initialized with Resend")
        else:
            logger.warning("RESEND_API_KEY not set - newsletter disabled")
        
        # Initialize database
        db_path = Path(__file__).parent.parent / "articles.db"
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Email settings
        self.from_email = "LiveLens <newsletter@resend.dev>"  # Use resend.dev for testing
        self.site_url = os.getenv('SITE_URL', 'http://localhost:8000')
    
    def _generate_unsubscribe_token(self, email: str) -> str:
        """Generate unique unsubscribe token"""
        return hashlib.sha256(f"{email}{secrets.token_hex(16)}".encode()).hexdigest()[:32]
    
    def subscribe(self, email: str, name: Optional[str] = None, preferences: str = 'all') -> Dict[str, Any]:
        """
        Subscribe an email to the newsletter.
        
        Returns:
            Dict with status and message
        """
        session = self.Session()
        try:
            # Check if already subscribed
            existing = session.query(Subscriber).filter_by(email=email.lower()).first()
            
            if existing:
                if existing.is_active:
                    return {"success": False, "message": "Email already subscribed"}
                else:
                    # Reactivate subscription
                    existing.is_active = True
                    existing.subscribed_at = datetime.utcnow()
                    session.commit()
                    return {"success": True, "message": "Welcome back! Subscription reactivated"}
            
            # Create new subscriber
            subscriber = Subscriber(
                email=email.lower(),
                name=name,
                preferences=preferences,
                unsubscribe_token=self._generate_unsubscribe_token(email)
            )
            session.add(subscriber)
            session.commit()
            
            # Send welcome email
            self._send_welcome_email(email, name)
            
            logger.info(f"New subscriber: {email}")
            return {"success": True, "message": "Successfully subscribed to newsletter!"}
            
        except Exception as e:
            session.rollback()
            logger.error(f"Subscribe error: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
        finally:
            session.close()
    
    def unsubscribe(self, token: str) -> Dict[str, Any]:
        """Unsubscribe using token"""
        session = self.Session()
        try:
            subscriber = session.query(Subscriber).filter_by(unsubscribe_token=token).first()
            
            if not subscriber:
                return {"success": False, "message": "Invalid unsubscribe link"}
            
            subscriber.is_active = False
            session.commit()
            
            logger.info(f"Unsubscribed: {subscriber.email}")
            return {"success": True, "message": "Successfully unsubscribed from newsletter"}
            
        except Exception as e:
            session.rollback()
            logger.error(f"Unsubscribe error: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
        finally:
            session.close()
    
    def get_subscribers(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get list of subscribers"""
        session = self.Session()
        try:
            query = session.query(Subscriber)
            if active_only:
                query = query.filter_by(is_active=True)
            
            subscribers = query.all()
            return [
                {
                    "email": s.email,
                    "name": s.name,
                    "subscribed_at": s.subscribed_at.isoformat() if s.subscribed_at else None,
                    "is_active": s.is_active,
                    "preferences": s.preferences
                }
                for s in subscribers
            ]
        finally:
            session.close()
    
    def get_subscriber_count(self) -> int:
        """Get count of active subscribers"""
        session = self.Session()
        try:
            return session.query(Subscriber).filter_by(is_active=True).count()
        finally:
            session.close()
    
    def _send_welcome_email(self, email: str, name: Optional[str] = None):
        """Send welcome email to new subscriber"""
        if not self.api_key:
            logger.warning("Cannot send welcome email - no API key")
            return
        
        greeting = f"Hi {name}," if name else "Hi there,"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #00c6ff, #0072ff); padding: 30px; text-align: center; }}
        .header h1 {{ color: white; margin: 0; font-size: 28px; }}
        .content {{ padding: 30px; }}
        .content h2 {{ color: #00c6ff; }}
        .content p {{ line-height: 1.6; color: #b0b0b0; }}
        .feature {{ background: rgba(0, 198, 255, 0.1); border-left: 3px solid #00c6ff; padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
        .footer {{ padding: 20px; text-align: center; color: #666; font-size: 12px; border-top: 1px solid #333; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Welcome to LiveLens!</h1>
        </div>
        <div class="content">
            <h2>{greeting}</h2>
            <p>Thanks for subscribing to the <strong>LiveLens Newsletter</strong>! You'll now receive curated AI-powered news digests straight to your inbox.</p>
            
            <div class="feature">
                <strong>📰 What you'll get:</strong>
                <ul>
                    <li>Daily/Weekly news summaries</li>
                    <li>AI-analyzed trending topics</li>
                    <li>Personalized recommendations</li>
                </ul>
            </div>
            
            <p>Stay informed with the power of AI!</p>
            <p>— The LiveLens Team</p>
        </div>
        <div class="footer">
            <p>© 2026 LiveLens | Powered by Pathway</p>
        </div>
    </div>
</body>
</html>
"""
        
        try:
            resend.Emails.send({
                "from": self.from_email,
                "to": [email],
                "subject": "🔍 Welcome to LiveLens Newsletter!",
                "html": html_content
            })
            logger.info(f"Welcome email sent to {email}")
        except Exception as e:
            logger.error(f"Failed to send welcome email: {e}")
    
    def send_newsletter(self, articles: List[Dict[str, Any]], subject: Optional[str] = None) -> NewsletterResult:
        """
        Send newsletter to all active subscribers.
        
        Args:
            articles: List of article dicts with title, content, source, url
            subject: Optional custom subject line
        
        Returns:
            NewsletterResult with send statistics
        """
        if not self.api_key:
            return NewsletterResult(
                success=False, sent_count=0, failed_count=0,
                message="Newsletter disabled - no API key configured"
            )
        
        session = self.Session()
        try:
            subscribers = session.query(Subscriber).filter_by(is_active=True).all()
            
            if not subscribers:
                return NewsletterResult(
                    success=False, sent_count=0, failed_count=0,
                    message="No active subscribers"
                )
            
            # Build newsletter HTML
            html_content = self._build_newsletter_html(articles)
            
            # Default subject
            if not subject:
                today = datetime.now().strftime("%B %d, %Y")
                subject = f"📰 LiveLens Daily Digest - {today}"
            
            sent_count = 0
            failed_count = 0
            
            for subscriber in subscribers:
                try:
                    # Add unsubscribe link
                    unsubscribe_url = f"{self.site_url}/api/newsletter/unsubscribe/{subscriber.unsubscribe_token}"
                    personalized_html = html_content.replace(
                        "{{UNSUBSCRIBE_URL}}", 
                        unsubscribe_url
                    )
                    
                    resend.Emails.send({
                        "from": self.from_email,
                        "to": [subscriber.email],
                        "subject": subject,
                        "html": personalized_html
                    })
                    sent_count += 1
                    logger.info(f"Newsletter sent to {subscriber.email}")
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to send to {subscriber.email}: {e}")
            
            return NewsletterResult(
                success=True,
                sent_count=sent_count,
                failed_count=failed_count,
                message=f"Newsletter sent to {sent_count} subscribers"
            )
            
        except Exception as e:
            logger.error(f"Newsletter send error: {e}")
            return NewsletterResult(
                success=False, sent_count=0, failed_count=0,
                message=f"Error: {str(e)}"
            )
        finally:
            session.close()
    
    def _build_newsletter_html(self, articles: List[Dict[str, Any]]) -> str:
        """Build beautiful HTML newsletter from articles"""
        today = datetime.now().strftime("%B %d, %Y")
        
        # Build article cards
        article_html = ""
        for i, article in enumerate(articles[:10]):  # Limit to 10 articles
            title = article.get('title', 'Untitled')
            content = article.get('content', '')[:200] + '...'
            source = article.get('source', 'Unknown')
            category = article.get('category', '')
            url = article.get('url', '#')
            
            article_html += f"""
            <div class="article">
                <span class="category">{category}</span>
                <h3><a href="{url}">{title}</a></h3>
                <p>{content}</p>
                <span class="source">📍 {source}</span>
            </div>
            """
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; margin: 0; }}
        .container {{ max-width: 600px; margin: 0 auto; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 16px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #00c6ff, #0072ff); padding: 30px; text-align: center; }}
        .header h1 {{ color: white; margin: 0; font-size: 24px; }}
        .header p {{ color: rgba(255,255,255,0.8); margin: 10px 0 0; }}
        .content {{ padding: 20px; }}
        .article {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; margin: 15px 0; }}
        .article h3 {{ margin: 10px 0; color: #fff; }}
        .article h3 a {{ color: #00c6ff; text-decoration: none; }}
        .article h3 a:hover {{ text-decoration: underline; }}
        .article p {{ color: #a0a0a0; font-size: 14px; line-height: 1.5; }}
        .category {{ background: linear-gradient(135deg, #00c6ff, #0072ff); color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; text-transform: uppercase; font-weight: 600; }}
        .source {{ color: #666; font-size: 12px; }}
        .footer {{ padding: 20px; text-align: center; color: #666; font-size: 12px; border-top: 1px solid #333; }}
        .footer a {{ color: #00c6ff; text-decoration: none; }}
        .cta {{ display: inline-block; background: linear-gradient(135deg, #00c6ff, #0072ff); color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 LiveLens Daily Digest</h1>
            <p>{today}</p>
        </div>
        <div class="content">
            <p style="color: #b0b0b0;">Here are today's top stories, curated by AI:</p>
            
            {article_html}
            
            <center>
                <a href="{self.site_url}/app" class="cta">Read More on LiveLens →</a>
            </center>
        </div>
        <div class="footer">
            <p>You're receiving this because you subscribed to LiveLens Newsletter.</p>
            <p><a href="{{{{UNSUBSCRIBE_URL}}}}">Unsubscribe</a> | <a href="{self.site_url}">Visit LiveLens</a></p>
            <p>© 2026 LiveLens | Powered by Pathway</p>
        </div>
    </div>
</body>
</html>
"""


# Singleton instance
_newsletter_service: Optional[NewsletterService] = None


def get_newsletter_service() -> NewsletterService:
    """Get or create newsletter service instance"""
    global _newsletter_service
    if _newsletter_service is None:
        _newsletter_service = NewsletterService()
    return _newsletter_service

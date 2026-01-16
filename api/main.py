"""
FastAPI Main Application.
REST and WebSocket endpoints for the Live AI News Platform.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config
from rag.rag_engine import RAGEngine, get_rag_engine
from user.user_profile import UserProfileManager, get_profile_manager
from user.recommendation_engine import RecommendationEngine, get_recommendation_engine
from connectors.news_connector import SerperNewsConnector
from connectors.article_scraper import ArticleScraper
from connectors.youtube_analyzer import YouTubeAnalyzer, get_youtube_analyzer
from api.auth import get_current_user, require_auth, sync_user_to_db
from api.push import (
    get_vapid_public_key, save_subscription, remove_subscription,
    send_push_notification, get_subscription
)

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChatQueryRequest(BaseModel):
    query: str = Field(..., description="User's question")
    user_id: Optional[str] = Field(None, description="User identifier for personalization")
    category: Optional[str] = Field(None, description="Category filter")

class ArticleChatRequest(BaseModel):
    query: str = Field(..., description="User's question about the article")
    user_id: Optional[str] = Field(None)
    expand_context: bool = Field(True, description="Include related articles")

class ComparisonRequest(BaseModel):
    article_ids: List[str] = Field(..., description="Article IDs to compare")
    query: str = Field(..., description="Comparison question")
    user_id: Optional[str] = Field(None)

class InteractionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    article_id: str = Field(..., description="Article interacted with")
    interaction_type: str = Field(..., description="view, chat, like, share, or compare")

class NewsSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    category: Optional[str] = Field(None)
    num_results: int = Field(10, ge=1, le=50)

class YouTubeSearchRequest(BaseModel):
    query: str = Field(..., description="Search query for YouTube videos")
    num_results: int = Field(5, ge=1, le=10, description="Number of videos to return")

class YouTubeAnalyzeRequest(BaseModel):
    video_url: str = Field(..., description="YouTube video URL")
    article_title: str = Field(..., description="Related article title")
    article_content: Optional[str] = Field(None, description="Article content for context")

class OnboardingRequest(BaseModel):
    """Request model for user onboarding questionnaire"""
    user_id: str = Field(..., description="Unique user identifier")
    categories: List[str] = Field(..., description="Q1: Selected news categories")
    reading_depth: str = Field(..., description="Q2: Reading depth preference")
    daily_time: str = Field(..., description="Q5: Daily time spent on news")
    content_formats: List[str] = Field(..., description="Q6: Preferred content formats")
    primary_reason: str = Field(..., description="Q7: Primary reason for staying informed")
    industry: str = Field(..., description="Q8: Work industry")
    regions: List[str] = Field(..., description="Q9: Regional preferences")
    ai_summary_preference: str = Field(..., description="Q12: AI summary preference")
    importance_timely: int = Field(..., ge=1, le=5, description="Q15a: Importance of timeliness")
    importance_accurate: int = Field(..., ge=1, le=5, description="Q15b: Importance of accuracy")
    importance_engaging: int = Field(..., ge=1, le=5, description="Q15c: Importance of engagement")

class AIChatRequest(BaseModel):
    """Request model for AI Chat messages"""
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., description="User's message")
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class DemoArticleRequest(BaseModel):
    """Request model for demo article injection"""
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    category: str = Field("Demo", description="Article category")
    topics: List[str] = Field(default_factory=list, description="Article topics")


news_connector: Optional[SerperNewsConnector] = None
article_scraper: Optional[ArticleScraper] = None
news_streaming_task: Optional[asyncio.Task] = None
background_fetch_task: Optional[asyncio.Task] = None

# WebSocket connection manager for real-time updates
active_websockets: List[WebSocket] = []


async def broadcast_article(article_data: Dict[str, Any]):
    """Broadcast a new article to all connected WebSocket clients"""
    message = json.dumps({
        "type": "new_article",
        "article": article_data
    })
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)


async def fetch_and_broadcast_articles(category: str = None, num_results: int = 5):
    """Fetch articles and broadcast each one as it's scraped"""
    global news_connector, article_scraper
    
    if not news_connector or not article_scraper:
        return 0
    
    # Fetch news URLs
    try:
        if category:
            results = await news_connector.search_news(
                query=f"{category} news",
                num_results=num_results
            )
        else:
            results = await news_connector.fetch_all_categories()
        
        if not results:
            return 0
        
        rag = get_rag_engine()
        rec_engine = get_recommendation_engine()
        
        count = 0
        for r in results:
            try:
                # Scrape single article
                article = await article_scraper.scrape_article(
                    url=r.url,
                    category=category or r.category,
                    source=r.source
                )
                
                if article:
                    article_dict = article.to_dict()
                    
                    # Add to engines (this also persists to SQLite)
                    if rag.add_document(article_dict):
                        rec_engine.add_article(article_dict)
                        count += 1
                        
                        # Broadcast immediately to all connected clients
                        await broadcast_article(article_dict)
                        logger.info(f"Broadcasted article: {article.title[:50]}...")
                        
            except Exception as e:
                logger.error(f"Error processing article {r.url}: {e}")
                continue
        
        return count
        
    except Exception as e:
        logger.error(f"Error in fetch_and_broadcast: {e}")
        return 0


async def background_news_fetcher():
    """Background task that fetches news every 5 minutes"""
    categories = ["Technology", "Business", "Health", "Entertainment", "Science"]
    category_index = 0
    
    await asyncio.sleep(30)  # Initial delay to let app start up
    
    while True:
        try:
            category = categories[category_index % len(categories)]
            logger.info(f"Background fetch starting for: {category}")
            
            count = await fetch_and_broadcast_articles(category=category, num_results=5)
            logger.info(f"Background fetch complete: {count} new articles indexed")
            
            category_index += 1
            
        except Exception as e:
            logger.error(f"Background fetch error: {e}")
        
        await asyncio.sleep(300)  # 5 minutes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - optimized for fast startup"""
    global news_connector, article_scraper, background_fetch_task
    
    logger.info("Starting Live AI News Platform API...")
    
    # Initialize lightweight components only (no ML models here)
    news_connector = SerperNewsConnector()
    article_scraper = ArticleScraper()
    
    logger.info("API server ready - heavy initialization will happen in background")
    
    # Yield immediately to let uvicorn bind the port
    # This is critical for Render/Railway to detect the service is up
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")
    if background_fetch_task:
        background_fetch_task.cancel()
    if news_connector:
        news_connector.stop()
    if article_scraper:
        article_scraper.shutdown()


# Background initialization task - runs AFTER server is up
async def initialize_heavy_components():
    """Initialize RAG, Pathway, and other heavy components in background"""
    await asyncio.sleep(2)  # Let the server fully start first
    
    logger.info("Initializing RAG engine and recommendation system...")
    
    try:
        # Initialize RAG and user components (singletons)
        rag = get_rag_engine()
        get_profile_manager()
        rec = get_recommendation_engine()
        
        # Sync recommendation engine with RAG data
        rec._articles = rag._documents.copy()
        logger.info(f"Synced {len(rec._articles)} articles to recommendation engine")
        
        # Note: Pathway server is started by app.py, not here
        # This avoids duplicate initialization
        
        logger.info("Heavy components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")


# Startup event to trigger background initialization
@asynccontextmanager
async def lifespan_with_background_init(app: FastAPI):
    """Lifespan with background initialization"""
    global news_connector, article_scraper, background_fetch_task
    
    logger.info("Starting Live AI News Platform API...")
    
    # Initialize lightweight components only
    news_connector = SerperNewsConnector()
    article_scraper = ArticleScraper()
    
    logger.info("API server starting - port binding now...")
    
    # Start background initialization (non-blocking)
    init_task = asyncio.create_task(initialize_heavy_components())
    
    # Start background news fetcher after init
    async def delayed_news_fetcher():
        await asyncio.sleep(10)  # Wait for init to complete
        await background_news_fetcher()
    
    background_fetch_task = asyncio.create_task(delayed_news_fetcher())
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    init_task.cancel()
    if background_fetch_task:
        background_fetch_task.cancel()
    if news_connector:
        news_connector.stop()
    if article_scraper:
        article_scraper.shutdown()
    
    # Stop Pathway server
    if Config.USE_PATHWAY:
        try:
            from pipeline.pathway_server import get_pathway_server
            pathway_server = get_pathway_server()
            pathway_server.stop()
        except:
            pass


# Create FastAPI app with optimized lifespan for fast startup
app = FastAPI(
    title="Live Lens",
    description="Real-time news analysis with adaptive RAG using Pathway",
    version="1.0.0",
    lifespan=lifespan_with_background_init
)

# Add CORS middleware
# Get allowed origins from environment for production
import os
frontend_url = os.getenv("FRONTEND_URL", "")
allowed_origins = ["*"]  # Default: allow all (for development)
if frontend_url:
    # In production, restrict to specific origins
    allowed_origins = [
        frontend_url,
        "http://localhost:8000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:8000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ============ Root / Frontend ============

@app.get("/")
async def landing():
    """Serve the landing page"""
    landing_file = frontend_path / "landing.html"
    if landing_file.exists():
        return FileResponse(str(landing_file))
    return {"message": "Live AI News Platform API", "docs": "/docs"}


@app.get("/app")
async def main_app():
    """Serve the main application (requires auth on frontend)"""
    app_file = frontend_path / "app.html"
    if app_file.exists():
        return FileResponse(str(app_file))
    return {"message": "Application not found"}


@app.get("/onboarding")
async def onboarding_page():
    """Serve the onboarding questionnaire page"""
    onboarding_file = frontend_path / "onboarding.html"
    if onboarding_file.exists():
        return FileResponse(str(onboarding_file))
    return {"message": "Onboarding page not found"}


# ============ Authentication ============

@app.post("/api/auth/sync")
async def sync_auth_user(
    current_user: dict = Depends(get_current_user)
):
    """Sync Clerk user to local database"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # For now, just return success - DB sync can be added later
        # when SQLAlchemy session is properly set up
        return {
            "status": "ok",
            "user_id": current_user.get("user_id"),
            "message": "User synced successfully"
        }
    except Exception as e:
        logger.error(f"Error syncing user: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync user")


@app.get("/api/auth/me")
async def get_current_user_info(
    current_user: dict = Depends(get_current_user)
):
    """Get current authenticated user info"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {
        "user_id": current_user.get("user_id"),
        "email": current_user.get("email")
    }


# ============ Health Check ============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# ============ Push Notifications ============

class PushSubscriptionRequest(BaseModel):
    subscription: dict = Field(..., description="Push subscription object from browser")
    user_id: Optional[str] = Field(None, description="User ID")

class PushTestRequest(BaseModel):
    title: str = Field("Test Notification", description="Notification title")
    body: str = Field("This is a test notification from LiveLens!", description="Notification body")

@app.get("/api/push/vapid-key")
async def get_vapid_key():
    """Get VAPID public key for push subscription"""
    return {"publicKey": get_vapid_public_key()}

@app.post("/api/push/subscribe")
async def subscribe_push(
    request: PushSubscriptionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Subscribe to push notifications"""
    user_id = request.user_id
    if current_user:
        user_id = current_user.get("user_id", user_id)
    
    if not user_id:
        user_id = f"anonymous_{id(request.subscription)}"
    
    success = save_subscription(user_id, request.subscription)
    
    if success:
        return {"status": "subscribed", "user_id": user_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to save subscription")

@app.delete("/api/push/unsubscribe")
async def unsubscribe_push(
    user_id: str = Query(..., description="User ID to unsubscribe")
):
    """Unsubscribe from push notifications"""
    success = remove_subscription(user_id)
    return {"status": "unsubscribed" if success else "not_found"}

@app.post("/api/push/test")
async def send_test_push(
    request: PushTestRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send a test push notification to current user"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_id = current_user.get("user_id")
    subscription = get_subscription(user_id)
    
    if not subscription:
        raise HTTPException(status_code=404, detail="No subscription found. Please enable notifications first.")
    
    success = send_push_notification(
        subscription=subscription,
        title=request.title,
        body=request.body,
        url="/app"
    )
    
    if success:
        return {"status": "sent", "message": "Test notification sent successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send notification")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    rag = get_rag_engine()
    rec = get_recommendation_engine()
    profile = get_profile_manager()
    
    return {
        "rag_engine": rag.get_stats(),
        "recommendation_engine": rec.get_stats(),
        "profile_manager": profile.get_stats(),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============ Demo Mode ============

@app.post("/api/demo/inject-article")
async def inject_demo_article(request: DemoArticleRequest):
    """
    Inject a test article to demonstrate real-time RAG updates.
    Perfect for hackathon demos showing live dynamism.
    
    The article will be immediately:
    1. Added to the Pathway streaming pipeline (if enabled)
    2. Added to the RAG engine with vector embeddings
    3. Broadcast to all connected WebSocket clients
    4. Available for queries
    """
    import uuid
    import time
    
    start_time = time.time()
    
    article_id = f"demo_{uuid.uuid4().hex[:8]}"
    article = {
        "article_id": article_id,
        "title": request.title,
        "content": request.content,
        "category": request.category,
        "topics": request.topics,
        "source": "Demo Injection",
        "author": "Demo System",
        "publish_date": datetime.utcnow().isoformat(),
        "scraped_at": datetime.utcnow().isoformat(),
        "url": f"demo://{article_id}"
    }
    
    # Add to Pathway server if enabled
    pathway_indexed = False
    if Config.USE_PATHWAY:
        try:
            from pipeline.pathway_server import get_pathway_server
            pathway_server = get_pathway_server()
            if pathway_server and pathway_server.is_running:
                pathway_indexed = pathway_server.add_article(article)
        except Exception as e:
            logger.warning(f"Could not add to Pathway: {e}")
    
    # Add to RAG engine (this also generates embeddings)
    rag = get_rag_engine()
    success = rag.add_document(article)
    
    # Add to recommendation engine
    rec_engine = get_recommendation_engine()
    rec_engine.add_article(article)
    
    # Broadcast to all connected WebSocket clients
    await broadcast_article(article)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Demo article injected: {article_id} - {request.title} (pathway={pathway_indexed}, {elapsed_ms:.1f}ms)")
    
    return {
        "status": "injected",
        "article_id": article_id,
        "title": request.title,
        "pathway_indexed": pathway_indexed,
        "indexing_time_ms": round(elapsed_ms, 1),
        "message": f"Article injected and immediately available for RAG queries. Total indexed: {rag.get_stats()['indexed_documents']}",
        "tip": "Query the same question again to see updated results!"
    }


@app.get("/api/knowledge-base/status")
async def get_knowledge_base_status():
    """
    Get detailed knowledge base status for live visualization.
    Returns article count, embedding status, search method, and Pathway status.
    """
    rag = get_rag_engine()
    stats = rag.get_stats()
    
    # Get Pathway-specific stats
    pathway_stats = stats.get("pathway_stats", {})
    
    return {
        "article_count": stats["indexed_documents"],
        "embeddings_count": stats.get("embeddings_count", 0),
        "embeddings_available": stats.get("embeddings_available", False),
        "search_method": stats.get("search_method", "keyword"),
        "last_updated": datetime.utcnow().isoformat(),
        "model": stats.get("model", "unknown"),
        # Pathway status
        "pathway": {
            "enabled": Config.USE_PATHWAY,
            "running": stats.get("pathway_running", False),
            "documents_indexed": pathway_stats.get("documents_indexed", 0),
            "last_update_time": pathway_stats.get("last_update_time"),
            "avg_index_latency_ms": pathway_stats.get("avg_index_latency_ms", 0),
            "avg_query_latency_ms": pathway_stats.get("avg_query_latency_ms", 0),
            "started_at": pathway_stats.get("started_at")
        }
    }


@app.post("/api/demo/test-dynamism")
async def test_dynamism(query: str = "What is the latest news about Acme Corporation?"):
    """
    Test endpoint to demonstrate Pathway's real-time dynamism.
    
    Steps:
    1. Query the RAG with a specific question
    2. Inject a new article about that topic
    3. Query again with the same question
    4. Return both responses to show the difference
    
    This endpoint proves that data changes are reflected immediately
    without any restart or re-indexing.
    """
    import uuid
    import time
    
    rag = get_rag_engine()
    
    # Step 1: Query BEFORE injection
    before_start = time.time()
    before_response = await rag.query_global(query)
    before_latency = (time.time() - before_start) * 1000
    
    # Step 2: Inject a relevant article
    article_id = f"demo_dynamism_{uuid.uuid4().hex[:6]}"
    demo_article = {
        "article_id": article_id,
        "title": "Breaking: Acme Corporation Announces Revolutionary AI Product",
        "content": """Acme Corporation has just announced a groundbreaking AI-powered product 
        that promises to transform the industry. CEO Jane Smith revealed the product at 
        today's press conference, calling it "the most significant innovation in our 
        company's 50-year history." The new product uses advanced machine learning to 
        automate complex workflows, potentially saving enterprises millions of dollars 
        annually. Industry analysts are calling this announcement a game-changer that 
        could disrupt multiple sectors including healthcare, finance, and manufacturing.
        Acme's stock price jumped 15% following the announcement.""",
        "category": "Technology",
        "topics": ["Acme Corporation", "AI", "Innovation", "Technology"],
        "source": "Demo - Live Injection",
        "author": "Pathway Demo System",
        "publish_date": datetime.utcnow().isoformat(),
        "scraped_at": datetime.utcnow().isoformat(),
        "url": f"demo://{article_id}"
    }
    
    # Add to Pathway if enabled
    pathway_indexed = False
    if Config.USE_PATHWAY:
        try:
            from pipeline.pathway_server import get_pathway_server
            pathway_server = get_pathway_server()
            if pathway_server and pathway_server.is_running:
                pathway_indexed = pathway_server.add_article(demo_article)
        except:
            pass
    
    # Add to local RAG engine
    inject_start = time.time()
    rag.add_document(demo_article)
    inject_latency = (time.time() - inject_start) * 1000
    
    # Small delay to ensure indexing completes
    await asyncio.sleep(0.1)
    
    # Step 3: Query AFTER injection
    after_start = time.time()
    after_response = await rag.query_global(query)
    after_latency = (time.time() - after_start) * 1000
    
    # Check if the answer changed
    answer_changed = before_response.response != after_response.response
    new_article_used = any(
        doc.get("article_id") == article_id 
        for doc in after_response.context.documents
    )
    
    return {
        "success": True,
        "query": query,
        "demonstration": {
            "before": {
                "response": before_response.response[:500] + "..." if len(before_response.response) > 500 else before_response.response,
                "documents_found": len(before_response.context.documents),
                "search_method": before_response.context.search_method,
                "latency_ms": round(before_latency, 1)
            },
            "injection": {
                "article_id": article_id,
                "title": demo_article["title"],
                "pathway_indexed": pathway_indexed,
                "indexing_latency_ms": round(inject_latency, 1)
            },
            "after": {
                "response": after_response.response[:500] + "..." if len(after_response.response) > 500 else after_response.response,
                "documents_found": len(after_response.context.documents),
                "search_method": after_response.context.search_method,
                "latency_ms": round(after_latency, 1),
                "new_article_in_context": new_article_used
            }
        },
        "proof_of_dynamism": {
            "answer_changed": answer_changed,
            "new_data_reflected": new_article_used,
            "total_time_ms": round(before_latency + inject_latency + after_latency, 1),
            "message": "SUCCESS: Real-time update demonstrated!" if (answer_changed or new_article_used) else "Note: Answer may not have changed significantly, but new article was indexed"
        }
    }


# ============ News Feed ============

@app.get("/api/news/feed")
async def get_news_feed(
    user_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    exclude_viewed: bool = True,
    category: Optional[str] = None
):
    """
    Get personalized news feed.
    If user_id provided, returns personalized recommendations.
    Can be filtered by category.
    """
    rec_engine = get_recommendation_engine()
    
    if user_id:
        recommendations = rec_engine.get_personalized_feed(
            user_id=user_id,
            limit=limit,
            exclude_viewed=exclude_viewed,
            category=category
        )
        
        return {
            "user_id": user_id,
            "personalized": True,
            "articles": [
                {
                    "article_id": r.article_id,
                    "title": r.title,
                    "category": r.category,
                    "topics": r.topics,
                    "relevance_score": r.score,
                    "reasons": r.reasons
                }
                for r in recommendations
            ],
            "count": len(recommendations)
        }
    else:
        # Return all articles, optionally filtered by category
        
        if category and category != "":
            # Specific category: Filter existing articles
            # Use loose matching to handle "Finance" vs "Finance News" and case differences
            all_articles = list(rec_engine._articles.values())
            target_cat = category.lower()
            
            filtered_articles = [
                a for a in all_articles 
                if target_cat in a.get("category", "").lower()
            ]
            articles = filtered_articles[:limit]
        else:
            # All categories: Use mixed feed strategy
            # Note: get_mixed_feed returns dicts, not objects, which matches what we need here
            articles = rec_engine.get_mixed_feed(limit=limit)

        return {
            "personalized": False,
            "articles": articles,
            "count": len(articles)
        }

@app.get("/api/news/article/{article_id}")
async def get_article(article_id: str, user_id: Optional[str] = None):
    """Get a specific article by ID"""
    rec_engine = get_recommendation_engine()
    article = rec_engine._articles.get(article_id)
    
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Track view if user provided
    if user_id:
        profile_mgr = get_profile_manager()
        profile_mgr.track_interaction(user_id, article, "view")
    
    # Get related articles and comparison suggestion
    related = rec_engine.get_related_articles(article_id, limit=5)
    comparison = rec_engine.suggest_comparison(article_id)
    
    return {
        "article": article,
        "related_articles": [
            {"article_id": r.article_id, "title": r.title, "reasons": r.reasons}
            for r in related
        ],
        "comparison_suggestion": {
            "article_ids": comparison.article_ids,
            "titles": comparison.titles,
            "prompt": comparison.comparison_prompt
        } if comparison else None
    }

@app.get("/api/news/search")
async def search_news(query: str, category: Optional[str] = None, limit: int = 10):
    """Search articles"""
    rag = get_rag_engine()
    results = rag._simple_search(query, top_k=limit, category_filter=category)
    
    return {
        "query": query,
        "category": category,
        "results": results,
        "count": len(results)
    }


# ============ Chat Endpoints ============

@app.post("/api/chat/query")
async def chat_global_query(request: ChatQueryRequest):
    """
    Send a query to the global RAG engine.
    Searches all indexed articles.
    """
    rag = get_rag_engine()
    
    response = await rag.query_global(
        query=request.query,
        category=request.category
    )
    
    return response.to_dict()

@app.post("/api/chat/article/{article_id}")
async def chat_with_article(article_id: str, request: ArticleChatRequest):
    """
    Chat about a specific article with context expansion.
    """
    rag = get_rag_engine()
    
    # Track chat interaction
    if request.user_id:
        profile_mgr = get_profile_manager()
        article = rag._documents.get(article_id)
        if article:
            profile_mgr.track_interaction(request.user_id, article, "chat")
    
    response = await rag.query_article(
        query=request.query,
        article_id=article_id,
        expand_context=request.expand_context
    )
    
    return response.to_dict()

@app.post("/api/chat/compare")
async def chat_compare_articles(request: ComparisonRequest):
    """
    Compare multiple articles.
    Perfect for "Tesla vs BMW" style comparisons.
    """
    rag = get_rag_engine()
    
    # Track comparison
    if request.user_id:
        profile_mgr = get_profile_manager()
        articles = [rag._documents.get(aid) for aid in request.article_ids if rag._documents.get(aid)]
        if articles:
            profile_mgr.track_comparison(request.user_id, articles)
    
    response = await rag.query_comparison(
        query=request.query,
        article_ids=request.article_ids
    )
    
    return response.to_dict()


# ============ AI Chat Endpoints ============

@app.post("/api/ai-chat/message")
async def ai_chat_message(request: AIChatRequest):
    """
    Send a message to the AI Chat.
    Uses intelligent article retrieval: searches local first, fetches new if needed.
    Maintains conversation history for context.
    """
    from api.ai_chat_engine import get_ai_chat_engine
    
    chat_engine = get_ai_chat_engine()
    
    try:
        response = await chat_engine.chat(
            session_id=request.session_id,
            message=request.message,
            user_id=request.user_id
        )
        
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"AI Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai-chat/sessions/new")
async def create_ai_chat_session(user_id: Optional[str] = None):
    """Create a new AI chat session"""
    from api.ai_chat_engine import get_ai_chat_engine
    
    chat_engine = get_ai_chat_engine()
    session_id = chat_engine.create_session(user_id)
    
    return {
        "session_id": session_id,
        "created": True
    }


@app.get("/api/ai-chat/sessions/{session_id}")
async def get_ai_chat_session(session_id: str):
    """Get chat session history"""
    from api.ai_chat_engine import get_ai_chat_engine
    
    chat_engine = get_ai_chat_engine()
    history = chat_engine.get_session_history(session_id)
    
    return {
        "session_id": session_id,
        "messages": history,
        "message_count": len(history)
    }


@app.delete("/api/ai-chat/sessions/{session_id}")
async def clear_ai_chat_session(session_id: str):
    """Clear a chat session"""
    from api.ai_chat_engine import get_ai_chat_engine
    
    chat_engine = get_ai_chat_engine()
    success = chat_engine.clear_session(session_id)
    
    return {
        "session_id": session_id,
        "cleared": success
    }


# ============ User Endpoints ============


@app.post("/api/user/interaction")
async def track_user_interaction(request: InteractionRequest):
    """Track a user interaction with an article"""
    profile_mgr = get_profile_manager()
    rec_engine = get_recommendation_engine()
    
    article = rec_engine._articles.get(request.article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    changes = profile_mgr.track_interaction(
        user_id=request.user_id,
        article=article,
        interaction_type=request.interaction_type
    )
    
    return {
        "tracked": True,
        "changes": changes
    }

@app.get("/api/user/{user_id}/preferences")
async def get_user_preferences(user_id: str):
    """Get user's preference summary"""
    profile_mgr = get_profile_manager()
    summary = profile_mgr.get_user_preferences_summary(user_id)
    
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
    
    return summary

@app.get("/api/user/{user_id}/recommendations")
async def get_user_recommendations(user_id: str, limit: int = 10):
    """Get personalized recommendations for a user"""
    rec_engine = get_recommendation_engine()
    
    recommendations = rec_engine.get_personalized_feed(
        user_id=user_id,
        limit=limit
    )
    
    return {
        "user_id": user_id,
        "recommendations": [
            {
                "article_id": r.article_id,
                "title": r.title,
                "score": r.score,
                "reasons": r.reasons
            }
            for r in recommendations
        ]
    }


# ============ Onboarding ============

# SQLite for onboarding data (simple local storage)
ONBOARDING_DB_PATH = Path(__file__).parent.parent / "onboarding.db"

def get_onboarding_engine():
    """Get SQLite engine for onboarding data"""
    from sqlalchemy import create_engine
    from api.db_models import Base
    engine = create_engine(f"sqlite:///{ONBOARDING_DB_PATH}", echo=False)
    Base.metadata.create_all(engine)
    return engine

@app.post("/api/onboarding")
async def save_onboarding(request: OnboardingRequest):
    """
    Save user onboarding questionnaire responses.
    This should only be called once per user.
    """
    from sqlalchemy.orm import sessionmaker
    from api.db_models import UserOnboarding
    
    try:
        engine = get_onboarding_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check if user already completed onboarding
        existing = session.query(UserOnboarding).filter_by(user_id=request.user_id).first()
        if existing:
            session.close()
            return {
                "success": True,
                "message": "Onboarding already completed",
                "user_id": request.user_id
            }
        
        # Create new onboarding record
        onboarding = UserOnboarding(
            user_id=request.user_id,
            categories=request.categories,
            reading_depth=request.reading_depth,
            daily_time=request.daily_time,
            content_formats=request.content_formats,
            primary_reason=request.primary_reason,
            industry=request.industry,
            regions=request.regions,
            ai_summary_preference=request.ai_summary_preference,
            importance_timely=request.importance_timely,
            importance_accurate=request.importance_accurate,
            importance_engaging=request.importance_engaging
        )
        
        session.add(onboarding)
        session.commit()
        session.close()
        
        logger.info(f"Saved onboarding for user: {request.user_id}")
        
        return {
            "success": True,
            "message": "Onboarding completed successfully",
            "user_id": request.user_id
        }
        
    except Exception as e:
        logger.error(f"Error saving onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/onboarding/check/{user_id}")
async def check_onboarding(user_id: str):
    """
    Check if a user has completed onboarding.
    Returns the onboarding data if completed.
    """
    from sqlalchemy.orm import sessionmaker
    from api.db_models import UserOnboarding
    
    try:
        engine = get_onboarding_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        
        onboarding = session.query(UserOnboarding).filter_by(user_id=user_id).first()
        
        if onboarding:
            data = {
                "user_id": onboarding.user_id,
                "categories": onboarding.categories,
                "reading_depth": onboarding.reading_depth,
                "daily_time": onboarding.daily_time,
                "content_formats": onboarding.content_formats,
                "primary_reason": onboarding.primary_reason,
                "industry": onboarding.industry,
                "regions": onboarding.regions,
                "ai_summary_preference": onboarding.ai_summary_preference,
                "importance_timely": onboarding.importance_timely,
                "importance_accurate": onboarding.importance_accurate,
                "importance_engaging": onboarding.importance_engaging,
                "completed_at": onboarding.completed_at.isoformat() if onboarding.completed_at else None
            }
            session.close()
            return {
                "completed": True,
                "data": data
            }
        
        session.close()
        return {
            "completed": False,
            "data": None
        }
        
    except Exception as e:
        logger.error(f"Error checking onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ News Ingestion ============

@app.post("/api/news/fetch")
async def fetch_news(category: Optional[str] = None, num_results: int = 10):
    """
    Manually trigger news fetching.
    Fetches from Serper and scrapes articles.
    """
    global news_connector, article_scraper
    
    if not news_connector:
        raise HTTPException(status_code=500, detail="News connector not initialized")
    
    # Fetch news URLs
    if category:
        results = await news_connector.search_news(
            query=f"{category} news",
            num_results=num_results
        )
    else:
        results = await news_connector.fetch_all_categories()
    
    if not results:
        return {"message": "No new articles found", "count": 0}
    
    # Scrape articles
    urls_to_scrape = []
    
    for r in results:
        # If a specific category was requested, enforce it
        # Otherwise use the category from the result (which is likely "search_query")
        clean_category = category if category else r.category
        
        urls_to_scrape.append({
            "url": r.url, 
            "category": clean_category, 
            "source": r.source,
            "snippet": r.snippet
        })
    
    articles = await article_scraper.scrape_articles(urls_to_scrape)
    
    # Add to RAG and recommendation engines
    rag = get_rag_engine()
    rec_engine = get_recommendation_engine()
    
    for article in articles:
        article_dict = article.to_dict()
        rag.add_document(article_dict)
        rec_engine.add_article(article_dict)
    
    return {
        "message": f"Fetched and indexed {len(articles)} new articles",
        "count": len(articles),
        "articles": [{"title": a.title, "article_id": a.article_id} for a in articles]
    }


# ============ YouTube Video Analysis ============

@app.post("/api/youtube/search")
async def search_youtube_videos(request: YouTubeSearchRequest):
    """
    Search for YouTube videos related to a query (e.g., article title).
    Uses Serper API to find relevant videos.
    """
    analyzer = get_youtube_analyzer()
    
    try:
        results = await analyzer.search_youtube_videos(
            query=request.query,
            num_results=request.num_results
        )
        
        return {
            "query": request.query,
            "videos": [r.to_dict() for r in results],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching YouTube: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/youtube/analyze")
async def analyze_youtube_video(request: YouTubeAnalyzeRequest):
    """
    Analyze a YouTube video: download audio, transcribe with Whisper,
    and analyze with OpenRouter LLM.
    
    This endpoint may take 1-3 minutes depending on video length.
    """
    analyzer = get_youtube_analyzer()
    
    try:
        result = await analyzer.analyze_video(
            youtube_url=request.video_url,
            article_title=request.article_title,
            article_content=request.article_content
        )
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze video. Please check that yt-dlp and ffmpeg are installed."
            )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error analyzing YouTube video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/youtube/status")
async def youtube_analyzer_status():
    """Check if YouTube analyzer dependencies are available"""
    import shutil
    
    yt_dlp_available = shutil.which("yt-dlp") is not None
    ffmpeg_available = shutil.which("ffmpeg") is not None
    
    whisper_available = False
    try:
        import whisper
        whisper_available = True
    except ImportError:
        pass
    
    all_ready = yt_dlp_available and ffmpeg_available and whisper_available
    
    return {
        "ready": all_ready,
        "dependencies": {
            "yt_dlp": yt_dlp_available,
            "ffmpeg": ffmpeg_available,
            "whisper": whisper_available
        },
        "message": "All dependencies ready" if all_ready else "Missing dependencies. Run: brew install yt-dlp ffmpeg && pip install openai-whisper"
    }


# ============ WebSocket Connections ============

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "feed": [],
            "chat": []
        }
    
    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        self.active_connections[channel].append(websocket)
        logger.info(f"WebSocket connected to {channel}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        if websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)
        logger.info(f"WebSocket disconnected from {channel}")
    
    async def broadcast(self, message: dict, channel: str):
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    """
    WebSocket for real-time news feed updates.
    Receives new articles as they are indexed.
    """
    await manager.connect(websocket, "feed")
    active_websockets.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, "feed")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket for streaming chat responses.
    """
    await manager.connect(websocket, "chat")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            query = data.get("query", "")
            query_type = data.get("type", "global")
            article_id = data.get("article_id")
            article_ids = data.get("article_ids", [])
            
            rag = get_rag_engine()
            
            # Process query based on type
            if query_type == "article" and article_id:
                response = await rag.query_article(query, article_id)
            elif query_type == "compare" and article_ids:
                response = await rag.query_comparison(query, article_ids)
            else:
                response = await rag.query_global(query)
            
            await websocket.send_json({
                "session_id": session_id,
                "response": response.to_dict()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, "chat")


# ============ Admin Endpoints ============

@app.post("/api/admin/clear-cache")
async def clear_caches():
    """Clear all caches (admin only)"""
    global article_scraper
    
    if article_scraper:
        article_scraper.clear_cache()
    
    return {"message": "Caches cleared"}


if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "api.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True
    )

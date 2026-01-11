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

# Global instances
news_connector: Optional[SerperNewsConnector] = None
article_scraper: Optional[ArticleScraper] = None
news_streaming_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global news_connector, article_scraper
    
    logger.info("Starting Live AI News Platform...")
    
    # Initialize components
    news_connector = SerperNewsConnector()
    article_scraper = ArticleScraper()
    
    # Initialize RAG and user components (singletons)
    get_rag_engine()
    get_profile_manager()
    get_recommendation_engine()
    
    logger.info("All components initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if news_connector:
        news_connector.stop()
    if article_scraper:
        article_scraper.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Live AI News Platform",
    description="Real-time news analysis with adaptive RAG using Pathway",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
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
async def root():
    """Serve the frontend"""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Live AI News Platform API", "docs": "/docs"}


# ============ Health Check ============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

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


# ============ News Feed ============

@app.get("/api/news/feed")
async def get_news_feed(
    user_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    exclude_viewed: bool = True
):
    """
    Get personalized news feed.
    If user_id provided, returns personalized recommendations.
    """
    rec_engine = get_recommendation_engine()
    
    if user_id:
        recommendations = rec_engine.get_personalized_feed(
            user_id=user_id,
            limit=limit,
            exclude_viewed=exclude_viewed
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
        # Return all articles without personalization
        articles = list(rec_engine._articles.values())[:limit]
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
    urls_to_scrape = [
        {"url": r.url, "category": r.category, "source": r.source}
        for r in results
    ]
    
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
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, "feed")


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

"""
News-Please FastAPI Service.
A dedicated microservice for article scraping using news-please.
Run separately: uvicorn scraper_service:app --port 8001
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from newsplease import NewsPlease

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News-Please Scraper Service",
    description="A microservice for extracting article content using news-please",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Browser-like headers
BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


# Request/Response models
class ScrapeRequest(BaseModel):
    url: str = Field(..., description="URL to scrape")
    timeout: int = Field(15, description="Request timeout in seconds")


class ScrapeResponse(BaseModel):
    success: bool
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    authors: Optional[List[str]] = None
    publish_date: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    language: Optional[str] = None
    source_domain: Optional[str] = None
    error: Optional[str] = None
    scraped_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class BatchScrapeRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to scrape")
    timeout: int = Field(15, description="Request timeout per URL")


class BatchScrapeResponse(BaseModel):
    total: int
    successful: int
    failed: int
    articles: List[ScrapeResponse]


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "news-please-scraper"}


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_article(request: ScrapeRequest):
    """
    Scrape a single article from URL.
    Returns extracted article data or error.
    """
    url = request.url
    logger.info(f"Scraping: {url}")
    
    try:
        # Use news-please with custom headers
        article = NewsPlease.from_url(
            url,
            request_args={
                'headers': BROWSER_HEADERS,
                'timeout': request.timeout
            }
        )
        
        # Check if we got a valid article
        if article is None:
            return ScrapeResponse(
                success=False,
                url=url,
                error="Could not extract article (returned None)"
            )
        
        # Handle dict responses (error cases)
        if isinstance(article, dict):
            return ScrapeResponse(
                success=False,
                url=url,
                error=f"Got dict response: {article.get('error', 'unknown error')}"
            )
        
        # Extract data safely
        title = getattr(article, 'title', None) or ""
        content = getattr(article, 'maintext', None) or getattr(article, 'text', None) or ""
        
        # Check for sufficient content
        if not title or not content or len(content) < 100:
            return ScrapeResponse(
                success=False,
                url=url,
                title=title if title else None,
                error=f"Insufficient content: title={bool(title)}, content_len={len(content)}"
            )
        
        # Extract publish date
        publish_date = None
        date_publish = getattr(article, 'date_publish', None)
        if date_publish:
            try:
                if hasattr(date_publish, 'isoformat'):
                    publish_date = date_publish.isoformat()
                else:
                    publish_date = str(date_publish)
            except Exception:
                pass
        
        # Get authors
        authors = getattr(article, 'authors', None) or []
        
        return ScrapeResponse(
            success=True,
            url=url,
            title=title,
            content=content,
            authors=authors if authors else None,
            publish_date=publish_date,
            description=getattr(article, 'description', None),
            image_url=getattr(article, 'image_url', None),
            language=getattr(article, 'language', None),
            source_domain=getattr(article, 'source_domain', None)
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error scraping {url}: {error_msg}")
        return ScrapeResponse(
            success=False,
            url=url,
            error=error_msg
        )


@app.post("/scrape/batch", response_model=BatchScrapeResponse)
async def scrape_batch(request: BatchScrapeRequest):
    """
    Scrape multiple articles from URLs.
    Returns list of results.
    """
    results = []
    successful = 0
    failed = 0
    
    for url in request.urls:
        try:
            result = await scrape_article(ScrapeRequest(url=url, timeout=request.timeout))
            results.append(result)
            if result.success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            results.append(ScrapeResponse(
                success=False,
                url=url,
                error=str(e)
            ))
            failed += 1
    
    return BatchScrapeResponse(
        total=len(request.urls),
        successful=successful,
        failed=failed,
        articles=results
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

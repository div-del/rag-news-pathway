"""
Article Scraper using news-please API service.
Extracts full article content from URLs via the scraper microservice.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import hashlib
from concurrent.futures import ThreadPoolExecutor

import httpx

from config import Config

logger = logging.getLogger(__name__)

# Scraper service configuration
SCRAPER_SERVICE_URL = "http://localhost:8001"


@dataclass
class ArticleContent:
    """Represents extracted article content"""
    article_id: str
    url: str
    title: str
    content: str
    author: Optional[str] = None
    publish_date: Optional[str] = None
    source: str = ""
    category: str = ""
    
    # Extracted metadata
    description: Optional[str] = None
    image_url: Optional[str] = None
    language: Optional[str] = None
    
    # Topics extracted from content (for recommendations)
    topics: List[str] = field(default_factory=list)
    
    # Timestamps
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_text_for_embedding(self) -> str:
        """Get combined text for vector embedding"""
        parts = [self.title]
        if self.description:
            parts.append(self.description)
        if self.content:
            parts.append(self.content)
        return "\n\n".join(parts)


class ArticleScraper:
    """
    Scrapes full article content from URLs using the scraper service API.
    Falls back to direct news-please if service is unavailable.
    """
    
    def __init__(self, max_workers: int = 5, service_url: str = None):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: Dict[str, ArticleContent] = {}
        self._service_url = service_url or SCRAPER_SERVICE_URL
        self._use_service = True  # Try to use service first
        self._http_client = httpx.AsyncClient(timeout=30.0)
        
    def _generate_article_id(self, url: str) -> str:
        """Generate unique article ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    def _extract_topics(self, title: str, content: str) -> List[str]:
        """
        Extract potential topics from article for recommendation.
        This is a simple keyword extraction - can be enhanced with NLP.
        """
        # Common topic keywords to look for
        topic_keywords = [
            # Technology
            "AI", "artificial intelligence", "machine learning", "technology",
            "startup", "tech", "software", "app", "cloud", "data",
            # Finance
            "stock", "market", "investment", "finance", "economy", "bank",
            "crypto", "bitcoin", "trading", "earnings",
            # Automotive
            "Tesla", "EV", "electric vehicle", "car", "automotive", 
            "BMW", "Ford", "Toyota", "Mercedes",
            # Business
            "CEO", "company", "business", "merger", "acquisition", "IPO",
            # Other
            "health", "science", "research", "climate", "energy", "space"
        ]
        
        text = f"{title} {content}".lower()
        topics = []
        
        for keyword in topic_keywords:
            if keyword.lower() in text:
                topics.append(keyword)
        
        return list(set(topics))[:10]  # Limit to 10 topics
    
    async def _scrape_via_service(self, url: str, category: str = "", source: str = "") -> Optional[ArticleContent]:
        """
        Scrape article via the scraper service API.
        """
        try:
            response = await self._http_client.post(
                f"{self._service_url}/scrape",
                json={"url": url, "timeout": 15}
            )
            
            if response.status_code != 200:
                logger.warning(f"Service returned {response.status_code} for {url}")
                return None
            
            data = response.json()
            
            if not data.get("success"):
                error = data.get("error", "Unknown error")
                logger.warning(f"Service failed for {url}: {error}")
                return None
            
            # Build ArticleContent from response
            title = data.get("title", "")
            content = data.get("content", "")
            
            if not title or not content:
                logger.warning(f"Empty content from service for {url}")
                return None
            
            article_id = self._generate_article_id(url)
            topics = self._extract_topics(title, content)
            
            authors = data.get("authors", [])
            author_str = ", ".join(authors) if authors else None
            
            extracted = ArticleContent(
                article_id=article_id,
                url=url,
                title=title,
                content=content,
                author=author_str,
                publish_date=data.get("publish_date"),
                source=source or data.get("source_domain", "") or "",
                category=category,
                description=data.get("description"),
                image_url=data.get("image_url"),
                language=data.get("language"),
                topics=topics,
                scraped_at=data.get("scraped_at", datetime.utcnow().isoformat())
            )
            
            logger.info(f"Scraped via service: {title[:50]}... ({len(content)} chars)")
            return extracted
            
        except httpx.ConnectError:
            logger.warning("Scraper service unavailable, falling back to direct scraping")
            self._use_service = False
            return await self._scrape_direct(url, category, source)
        except Exception as e:
            logger.error(f"Service error for {url}: {e}")
            return None
    
    async def _scrape_soup_fallback(self, url: str) -> Optional[dict]:
        """Fallback to simple BeautifulSoup scraping"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }
            
            async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
                response = await client.get(url, headers=headers, timeout=15.0)
                
            if response.status_code != 200:
                return None
                
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer", "header", "iframe"]):
                script.decompose()
                
            title = soup.title.string if soup.title else ""
            if not title:
                h1 = soup.find('h1')
                title = h1.get_text().strip() if h1 else ""
                
            # Get text from paragraphs
            paragraphs = soup.find_all('p')
            text_content = "\n\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40])
            
            if len(text_content) < 100:
                return None
                
            return {
                "title": title.strip(),
                "content": text_content,
                "description": "", 
                "source_domain": url.split("//")[-1].split("/")[0]
            }
            
        except Exception as e:
            logger.error(f"Soup fallback failed for {url}: {e}")
            return None

    async def _scrape_direct(self, url: str, category: str = "", source: str = "") -> Optional[ArticleContent]:
        """
        Direct scraping using news-please (fallback).
        """
        title = ""
        content = ""
        authors = None
        publish_date = None
        description = None
        image_url = None
        source_domain = ""

        # Try news-please first
        try:
            from newsplease import NewsPlease
            
            # Browser-like headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml',
            }
            
            loop = asyncio.get_event_loop()
            article = await loop.run_in_executor(
                self._executor,
                lambda: NewsPlease.from_url(url, request_args={'headers': headers, 'timeout': 15})
            )
            
            if article and not isinstance(article, dict):
                title = getattr(article, 'title', None) or ""
                content = getattr(article, 'maintext', None) or ""
                authors = getattr(article, 'authors', None)
                description = getattr(article, 'description', None)
                image_url = getattr(article, 'image_url', None)
                source_domain = getattr(article, 'source_domain', "") or ""
                if getattr(article, 'date_publish', None):
                    try:
                        publish_date = article.date_publish.isoformat()
                    except:
                        pass
        except Exception as e:
            logger.warning(f"News-please failed for {url}: {e}")

        # If news-please failed or returned empty content, try soup fallback
        if not title or not content or len(content) < 100:
            logger.info(f"News-please yielded insufficient content for {url}, trying soup fallback")
            soup_data = await self._scrape_soup_fallback(url)
            if soup_data:
                title = soup_data.get("title") or title
                content = soup_data.get("content") or content
                source_domain = soup_data.get("source_domain") or source_domain
                description = soup_data.get("description") or description

        # Final check
        if not title or not content or len(content) < 100:
            logger.warning(f"Insufficient content from all methods: {url}")
            return None
            
        article_id = self._generate_article_id(url)
        topics = self._extract_topics(title, content)
        author_str = ", ".join(authors) if authors else None
        
        return ArticleContent(
            article_id=article_id,
            url=url,
            title=title,
            content=content,
            author=author_str,
            publish_date=publish_date,
            source=source or source_domain or "",
            category=category,
            description=description,
            image_url=image_url,
            language=None,
            topics=topics
        )

    
    async def scrape_article(
        self,
        url: str,
        category: str = "",
        source: str = ""
    ) -> Optional[ArticleContent]:
        """
        Asynchronously scrape a single article.
        
        Args:
            url: Article URL to scrape
            category: News category (for metadata)
            source: Source name (for metadata)
        
        Returns:
            ArticleContent if successful, None otherwise
        """
        # Check cache first
        article_id = self._generate_article_id(url)
        if article_id in self._cache:
            logger.debug(f"Cache hit for: {url}")
            return self._cache[article_id]
        
        # Try service first, fall back to direct scraping
        if self._use_service:
            result = await self._scrape_via_service(url, category, source)
        else:
            result = await self._scrape_direct(url, category, source)
        
        # Cache successful results
        if result:
            self._cache[article_id] = result
        
        return result
    
    async def scrape_articles(
        self,
        urls: List[Dict[str, str]],
        batch_size: int = 5
    ) -> List[ArticleContent]:
        """
        Scrape multiple articles concurrently.
        
        Args:
            urls: List of dicts with 'url', 'category', 'source' keys
            batch_size: Number of concurrent scrapes
        
        Returns:
            List of successfully scraped ArticleContent objects
        """
        results = []
        
        # Process in batches
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            
            tasks = [
                self.scrape_article(
                    item.get("url", ""),
                    item.get("category", ""),
                    item.get("source", "")
                )
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            
            for result in batch_results:
                if result:
                    results.append(result)
            
            # Small delay between batches
            if i + batch_size < len(urls):
                await asyncio.sleep(1)
        
        logger.info(f"Scraped {len(results)} of {len(urls)} articles")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics"""
        return {
            "cached_articles": len(self._cache),
            "max_workers": self.max_workers
        }
    
    def clear_cache(self):
        """Clear the article cache"""
        self._cache.clear()
        logger.info("Article cache cleared")
    
    def shutdown(self):
        """Shutdown the thread pool"""
        self._executor.shutdown(wait=True)
        logger.info("Article scraper shutdown complete")


# Utility function for testing
async def test_scraper():
    """Test the article scraper"""
    scraper = ArticleScraper()
    
    # Test URLs (replace with actual news URLs)
    test_urls = [
        {
            "url": "https://www.bbc.com/news",
            "category": "Technology",
            "source": "BBC"
        }
    ]
    
    print("Testing Article Scraper...")
    
    if test_urls:
        results = await scraper.scrape_articles(test_urls)
        
        for article in results:
            print(f"\nTitle: {article.title}")
            print(f"Content length: {len(article.content)} chars")
            print(f"Topics: {article.topics}")
    
    scraper.shutdown()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_scraper())

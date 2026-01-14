"""
Serper API News Connector.
Fetches news URLs from Serper API based on categories and search queries.
Designed to work as a streaming data source for Pathway.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
import httpx

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class NewsSearchResult:
    """Represents a single news search result"""
    url: str
    title: str
    snippet: str
    source: str
    date: Optional[str]
    category: str
    search_query: str
    fetched_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SerperNewsConnector:
    """
    Connects to Serper API to fetch news URLs.
    Implements streaming pattern for continuous news discovery.
    """
    
    def __init__(
        self,
        api_key: str = None,
        categories: List[str] = None,
        fetch_interval_seconds: int = None
    ):
        self.api_key = api_key or Config.SERPER_API_KEY
        self.categories = categories or Config.NEWS_CATEGORIES
        self.fetch_interval = fetch_interval_seconds or Config.NEWS_FETCH_INTERVAL_SECONDS
        self.base_url = "https://google.serper.dev/news"
        
        # Track seen URLs to avoid duplicates
        self._seen_urls: set = set()
        self._running = False
        
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is required")
    
    async def search_news(
        self,
        query: str,
        num_results: int = 10,
        time_period: str = "d"  # d=day, w=week, m=month
    ) -> List[NewsSearchResult]:
        """
        Search for news articles using Serper API.
        
        Args:
            query: Search query (e.g., category name or specific topic)
            num_results: Number of results to fetch (max 100)
            time_period: Time period filter (d=day, w=week, m=month)
        
        Returns:
            List of NewsSearchResult objects
        """
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results,
            "tbs": f"qdr:{time_period}"  # Time-based search
        }
        
        results = []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                news_items = data.get("news", [])
                logger.info(f"Fetched {len(news_items)} news items for query: {query}")
                
                for item in news_items:
                    url = item.get("link", "")
                    
                    # Skip if already seen
                    if url in self._seen_urls:
                        continue
                    
                    self._seen_urls.add(url)
                    
                    result = NewsSearchResult(
                        url=url,
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        source=item.get("source", ""),
                        date=item.get("date"),
                        category=query,
                        search_query=query,
                        fetched_at=datetime.utcnow().isoformat()
                    )
                    results.append(result)
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching news for '{query}': {e}")
        except Exception as e:
            logger.error(f"Error fetching news for '{query}': {e}")
        
        return results
    
    async def fetch_all_categories(self) -> List[NewsSearchResult]:
        """Fetch news from all configured categories"""
        all_results = []
        
        for category in self.categories:
            results = await self.search_news(
                query=f"{category} news",
                num_results=10,
                time_period="d"  # Last 24 hours
            )
            all_results.extend(results)
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.5)
        
        logger.info(f"Total new articles discovered: {len(all_results)}")
        return all_results
    
    async def stream_news(self) -> AsyncGenerator[NewsSearchResult, None]:
        """
        Continuously stream news results.
        This is the main entry point for Pathway integration.
        
        Yields:
            NewsSearchResult objects as they are discovered
        """
        self._running = True
        logger.info(f"Starting news stream with {len(self.categories)} categories")
        logger.info(f"Fetch interval: {self.fetch_interval} seconds")
        
        while self._running:
            try:
                # Fetch from all categories
                results = await self.fetch_all_categories()
                
                # Yield each result
                for result in results:
                    yield result
                
                # Wait before next fetch cycle
                logger.info(f"Waiting {self.fetch_interval}s before next fetch...")
                await asyncio.sleep(self.fetch_interval)
                
            except asyncio.CancelledError:
                logger.info("News stream cancelled")
                break
            except Exception as e:
                logger.error(f"Error in news stream: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    def stop(self):
        """Stop the streaming"""
        self._running = False
        logger.info("News stream stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics"""
        return {
            "categories": self.categories,
            "fetch_interval_seconds": self.fetch_interval,
            "total_urls_seen": len(self._seen_urls),
            "is_running": self._running
        }


# Utility function for quick testing
async def test_connector():
    """Test the Serper connector"""
    connector = SerperNewsConnector()
    
    print("Testing Serper News Connector...")
    print(f"Categories: {connector.categories}")
    
    # Fetch one category
    results = await connector.search_news("Technology news", num_results=5)
    
    print(f"\nFound {len(results)} articles:")
    for result in results:
        print(f"  - {result.title[:50]}... ({result.source})")
        print(f"    URL: {result.url[:80]}...")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_connector())

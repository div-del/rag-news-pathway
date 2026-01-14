"""
Live AI News Platform - Main Application Entry Point.
Orchestrates all components: Pipeline, API Server, and News Streaming.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsAIPlatform:
    """
    Main application class that orchestrates all components.
    """
    
    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._news_task: Optional[asyncio.Task] = None
        
    async def start_news_streaming(self):
        """Start continuous news fetching in background"""
        from connectors.news_connector import SerperNewsConnector
        from connectors.article_scraper import ArticleScraper
        from rag.rag_engine import get_rag_engine
        from user.recommendation_engine import get_recommendation_engine
        
        connector = SerperNewsConnector()
        scraper = ArticleScraper()
        rag = get_rag_engine()
        rec_engine = get_recommendation_engine()
        
        logger.info("Starting news streaming...")
        
        try:
            async for news_item in connector.stream_news():
                if self._shutdown_event.is_set():
                    break
                
                # Scrape the article
                article = await scraper.scrape_article(
                    url=news_item.url,
                    category=news_item.category,
                    source=news_item.source
                )
                
                if article:
                    article_dict = article.to_dict()
                    
                    # Add to engines
                    rag.add_document(article_dict)
                    rec_engine.add_article(article_dict)
                    
                    logger.info(f"Indexed: {article.title[:50]}...")
                    
        except asyncio.CancelledError:
            logger.info("News streaming cancelled")
        finally:
            connector.stop()
            scraper.shutdown()
    
    async def start_api_server(self):
        """Start the FastAPI server"""
        import uvicorn
        
        config = uvicorn.Config(
            "api.main:app",
            host=Config.HOST,
            port=Config.PORT,
            log_level="info",
            reload=False  # Disable reload in production
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"Starting API server on {Config.HOST}:{Config.PORT}")
        
        await server.serve()
    
    async def run(self, enable_news_streaming: bool = True):
        """
        Run the complete platform.
        
        Args:
            enable_news_streaming: Whether to start background news fetching
        """
        logger.info("=" * 50)
        logger.info("Live AI News Platform Starting...")
        logger.info("=" * 50)
        logger.info(f"Host: {Config.HOST}")
        logger.info(f"Port: {Config.PORT}")
        logger.info(f"News Categories: {Config.NEWS_CATEGORIES}")
        logger.info(f"LLM Model: {Config.LLM_MODEL}")
        logger.info("=" * 50)
        
        tasks = []
        
        # Start API server
        api_task = asyncio.create_task(self.start_api_server())
        tasks.append(api_task)
        
        # Start news streaming if enabled
        if enable_news_streaming:
            self._news_task = asyncio.create_task(self.start_news_streaming())
            tasks.append(self._news_task)
        
        # Wait for shutdown or completion
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Platform shutdown requested")
        
        logger.info("Platform stopped")
    
    def shutdown(self):
        """Signal shutdown to all components"""
        self._shutdown_event.set()
        if self._news_task:
            self._news_task.cancel()


def main():
    """Main entry point"""
    platform = NewsAIPlatform()
    
    # Handle signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        platform.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Live AI News Platform")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable automatic news streaming"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=Config.PORT,
        help=f"API server port (default: {Config.PORT})"
    )
    
    args = parser.parse_args()
    
    # Override port if specified
    if args.port != Config.PORT:
        Config.PORT = args.port
    
    # Run the platform
    asyncio.run(platform.run(enable_news_streaming=not args.no_streaming))


if __name__ == "__main__":
    main()

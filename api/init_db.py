"""
Database initialization script.
Run this to create all tables in the PostgreSQL database.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db_utils import init_db_manager
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_database():
    """Initialize the database and create all tables"""
    logger.info("Starting database initialization...")
    logger.info(f"Connection string: {Config.POSTGRES_CONNECTION_STRING}")
    
    try:
        # Initialize database manager
        db_manager = init_db_manager(Config.POSTGRES_CONNECTION_STRING, use_async=True)
        
        # Create all tables
        await db_manager.create_tables()
        
        logger.info("✅ Database initialized successfully!")
        logger.info("Tables created:")
        logger.info("  - users")
        logger.info("  - user_preferences")
        logger.info("  - articles")
        logger.info("  - user_interactions")
        logger.info("  - article_comparisons")
        
        await db_manager.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(initialize_database())

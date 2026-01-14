"""
Database connection management.
"""
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import Config
from api.db_models import Base

logger = logging.getLogger(__name__)

# Create engine
# Use connection pooling to handle multiple requests handles
engine = create_engine(
    Config.POSTGRES_CONNECTION_STRING, 
    echo=False,
    pool_size=Config.DB_POOL_SIZE,
    max_overflow=Config.DB_MAX_OVERFLOW
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized/verified")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        # Don't raise here to allow app to start even if DB is temporarily flaky,
        # but in production we might want to fail hard.
        pass

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get database session context manager"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()

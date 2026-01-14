"""
Database utility functions for connection management and operations.
"""

import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool
import logging

from config import Config
from api.db_models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, connection_string: str, use_async: bool = True):
        self.connection_string = connection_string
        self.use_async = use_async
        
        if use_async:
            # Convert postgresql:// to postgresql+asyncpg://
            async_conn_string = connection_string.replace(
                'postgresql://', 'postgresql+asyncpg://'
            )
            self.engine = create_async_engine(
                async_conn_string,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
            self.SessionLocal = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        else:
            # Synchronous engine for migrations
            self.engine = create_engine(
                connection_string,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )
    
    async def create_tables(self):
        """Create all tables in the database"""
        if self.use_async:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully (async)")
        else:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully (sync)")
    
    async def drop_tables(self):
        """Drop all tables (use with caution!)"""
        if self.use_async:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.warning("All database tables dropped (async)")
        else:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped (sync)")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        if not self.use_async:
            raise RuntimeError("Cannot get async session from sync engine")
        
        async with self.SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    def get_sync_session(self) -> Session:
        """Get synchronous database session"""
        if self.use_async:
            raise RuntimeError("Cannot get sync session from async engine")
        return self.SessionLocal()
    
    async def close(self):
        """Close database connections"""
        if self.use_async:
            await self.engine.dispose()
        else:
            self.engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def init_db_manager(connection_string: str = None, use_async: bool = True) -> DatabaseManager:
    """Initialize the global database manager"""
    global db_manager
    
    if connection_string is None:
        connection_string = Config.POSTGRES_CONNECTION_STRING
    
    db_manager = DatabaseManager(connection_string, use_async=use_async)
    logger.info(f"Database manager initialized ({'async' if use_async else 'sync'})")
    return db_manager


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    if db_manager is None:
        raise RuntimeError("Database manager not initialized. Call init_db_manager() first.")
    return db_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database sessions"""
    manager = get_db_manager()
    async with manager.get_session() as session:
        yield session


# Utility functions for common operations

async def ensure_user_exists(session: AsyncSession, user_id: str):
    """Ensure a user exists in the database, create if not"""
    from sqlalchemy import select
    from api.db_models import User
    
    result = await session.execute(
        select(User).where(User.user_id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        user = User(user_id=user_id)
        session.add(user)
        await session.flush()
        logger.info(f"Created new user: {user_id}")
    
    return user


async def update_user_preference(
    session: AsyncSession,
    user_id: str,
    preference_type: str,
    preference_value: str,
    score_delta: float = 0.1
):
    """Update or create user preference with incremental scoring"""
    from sqlalchemy import select
    from api.db_models import UserPreference
    
    # Ensure user exists
    await ensure_user_exists(session, user_id)
    
    # Get or create preference
    result = await session.execute(
        select(UserPreference).where(
            UserPreference.user_id == user_id,
            UserPreference.preference_type == preference_type,
            UserPreference.preference_value == preference_value
        )
    )
    pref = result.scalar_one_or_none()
    
    if pref is None:
        pref = UserPreference(
            user_id=user_id,
            preference_type=preference_type,
            preference_value=preference_value,
            score=0.5 + score_delta
        )
        session.add(pref)
        logger.debug(f"Created preference: {user_id} -> {preference_value}")
    else:
        # Increment score with decay (max 1.0)
        pref.score = min(1.0, pref.score + score_delta)
        logger.debug(f"Updated preference: {user_id} -> {preference_value} (score: {pref.score:.2f})")
    
    await session.flush()
    return pref

"""
Database Connection and Session Management

Handles:
- Creating database engine
- Managing sessions
- Creating tables
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from .config import DatabaseConfig, get_settings
from .models import Base


class Database:
    """
    Database connection manager.

    Usage:
        db = Database()
        db.create_tables()

        with db.get_session() as session:
            # Do database operations
            session.add(video)
            session.commit()
    """

    def __init__(self, database_url: str = None):
        """
        Initialize database connection.

        Args:
            database_url: Override default connection URL
        """
        if database_url is None:
            database_url = DatabaseConfig.get_database_url()

        if "sqlite" in database_url:
            self.engine = create_engine(database_url, echo=False)
        else:
            s = get_settings()
            try:
                self.engine = create_engine(
                    database_url,
                    pool_size=s.DB_POOL_SIZE,
                    max_overflow=s.DB_MAX_OVERFLOW,
                    echo=False
                )
            except (ModuleNotFoundError, ImportError) as e:
                import logging
                logging.getLogger(__name__).warning(
                    "PostgreSQL driver unavailable (%s). Falling back to SQLite development database.", e
                )
                self.engine = create_engine("sqlite:///data/video_recommender.db", echo=False)

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def create_tables(self):
        """Create all tables defined in models."""
        Base.metadata.create_all(bind=self.engine)
        print("Database tables created successfully")

    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        print("Database tables dropped")

    def init_pgvector(self):
        """Initialize pgvector extension if running on PostgreSQL."""
        if "postgresql" not in str(self.engine.url):
            return
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("pgvector extension enabled")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.

        Usage:
            with db.get_session() as session:
                session.query(Video).all()
        """
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def get_session_direct(self) -> Session:
        """
        Get a session directly (caller must close it).

        Use get_session() context manager when possible.
        """
        return self.SessionLocal()


# Global database instance
_db: Database = None


def get_database() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def init_database():
    """Initialize database with tables and extensions."""
    db = get_database()

    # Enable pgvector extension
    try:
        db.init_pgvector()
    except Exception as e:
        print(f"Could not enable pgvector: {e}")
        print("Continuing without vector extension...")

    # Create tables
    db.create_tables()

    return db


# FastAPI dependency
def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    Usage in FastAPI:
        @app.get("/videos")
        def get_videos(session: Session = Depends(get_db_session)):
            return session.query(Video).all()
    """
    db = get_database()
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()

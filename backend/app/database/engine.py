"""
Database session and engine configuration.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from app.core.config import get_settings

settings = get_settings()

# SQLAlchemy Base for all models
Base = declarative_base()

# Create database engine
if settings.database_url.startswith("sqlite"):
    # SQLite requires special handling for threading
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        echo=settings.debug,
        future=True,
    )
else:
    # PostgreSQL, MySQL, etc.
    engine = create_engine(
        settings.database_url,
        echo=settings.debug,
        future=True,
        pool_pre_ping=True,  # Test connections before use
    )

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)


def get_db() -> Session:
    """
    Dependency to get database session in route handlers.
    
    Usage:
        def get_alerts(db: Session = Depends(get_db)):
            return db.query(Alert).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)

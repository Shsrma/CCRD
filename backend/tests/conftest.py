"""
Test configuration and fixtures for pytest.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os
import sys
from pathlib import Path

# Ensure backend and project root are in path
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.database.engine import Base, get_db
from app.core.security import hash_password
from app.models import User


# Use in-memory SQLite with StaticPool for fast, isolated tests without file locking
from sqlalchemy.pool import StaticPool

@pytest.fixture(scope="function")
def test_db():
    """Create a fresh in-memory test database for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def client(test_db):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def test_user(test_db):
    """Create a test user in the database."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=hash_password("TestPass123"),
        role="fraud_officer",
        is_active=True,
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def auth_token(client, test_user):
    """Get authentication token for test user."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "username": "testuser",
            "password": "TestPass123",
        }
    )
    assert response.status_code == 200
    return response.json()["access_token"]

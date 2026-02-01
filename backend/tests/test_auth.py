"""
Unit tests for authentication routes.
"""

import pytest
from fastapi import status


class TestUserRegistration:
    """Test user registration endpoint."""
    
    def test_signup_success(self, client, test_db):
        """Test successful user registration."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "newuser",
                "email": "new@example.com",
                "password": "SecurePass123",
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["username"] == "newuser"
        assert data["role"] == "fraud_officer"
        assert "hashed_password" not in data  # Password shouldn't be exposed
    
    def test_signup_duplicate_username(self, client, test_user):
        """Test signup with existing username."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "testuser",  # Already exists
                "email": "other@example.com",
                "password": "SecurePass123",
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already exists" in response.json()["detail"]
    
    def test_signup_weak_password(self, client):
        """Test signup with weak password."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "weakpassuser",
                "email": "weak@example.com",
                "password": "weakpass",  # No uppercase or digits
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_signup_short_username(self, client):
        """Test signup with username too short."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "ab",  # Too short
                "password": "SecurePass123",
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestLogin:
    """Test login endpoint."""
    
    def test_login_success(self, client, test_user):
        """Test successful login."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "testuser",
                "password": "TestPass123",
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
    
    def test_login_wrong_password(self, client, test_user):
        """Test login with wrong password."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "testuser",
                "password": "WrongPassword123",
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect" in response.json()["detail"]
    
    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "nonexistent",
                "password": "AnyPassword123",
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestTokenValidation:
    """Test JWT token validation."""
    
    def test_protected_route_with_token(self, client, auth_token):
        """Test accessing protected route with valid token."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/api/v1/alerts/", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_protected_route_without_token(self, client):
        """Test accessing protected route without token."""
        response = client.get("/api/v1/alerts/")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_protected_route_with_invalid_token(self, client):
        """Test accessing protected route with invalid token."""
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = client.get("/api/v1/alerts/", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

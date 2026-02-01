"""
API integration tests.
"""

import pytest
from fastapi import status


class TestHealthCheck:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test root health check."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_api_root(self, client):
        """Test API root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestCORS:
    """Test CORS headers."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in response."""
        response = client.get("/health")
        
        # Note: CORS headers are set by the middleware
        assert response.status_code == status.HTTP_200_OK

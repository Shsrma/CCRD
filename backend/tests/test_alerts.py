"""
Integration tests for alerts functionality.
"""

import pytest
from fastapi import status


class TestFraudAlerts:
    """Test fraud alert management."""
    
    def test_get_alerts(self, client, auth_token):
        """Test retrieving alerts."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        response = client.get(
            "/api/v1/alerts/",
            headers=headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.json(), list)
    
    def test_get_pending_alerts_count(self, client, auth_token):
        """Test getting count of pending alerts."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        response = client.get(
            "/api/v1/alerts/pending/count",
            headers=headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "pending_count" in data
        assert isinstance(data["pending_count"], int)
    
    def test_filter_alerts_by_status(self, client, auth_token):
        """Test filtering alerts by status."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        response = client.get(
            "/api/v1/alerts/?alert_status=pending",
            headers=headers
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_invalid_alert_status_filter(self, client, auth_token):
        """Test filtering with invalid status."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        response = client.get(
            "/api/v1/alerts/?alert_status=invalid_status",
            headers=headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

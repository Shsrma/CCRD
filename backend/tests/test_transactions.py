"""
Unit tests for transaction prediction endpoints.
"""

import pytest
import numpy as np
from fastapi import status


class TestTransactionPrediction:
    """Test fraud prediction endpoint."""
    
    def test_predict_legitimate_transaction(self, client, auth_token):
        """Test prediction for a legitimate transaction."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        response = client.post(
            "/api/v1/transactions/predict",
            json={
                "amount": 50.00,
                "timestamp": 1704067200.0,
                "features": [0.1, -0.5, 0.3, 0.2, -0.1]
            },
            headers=headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "transaction_id" in data
        assert "fraud_prediction" in data
        assert data["fraud_prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
    
    def test_predict_without_authentication(self, client):
        """Test prediction without authentication."""
        response = client.post(
            "/api/v1/transactions/predict",
            json={
                "amount": 50.00,
                "timestamp": 1704067200.0,
                "features": [0.1, -0.5, 0.3, 0.2]
            }
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_predict_invalid_amount(self, client, auth_token):
        """Test prediction with invalid amount."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        response = client.post(
            "/api/v1/transactions/predict",
            json={
                "amount": -50.00,  # Negative amount
                "timestamp": 1704067200.0,
                "features": [0.1, -0.5, 0.3, 0.2]
            },
            headers=headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_missing_features(self, client, auth_token):
        """Test prediction with missing features."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        response = client.post(
            "/api/v1/transactions/predict",
            json={
                "amount": 50.00,
                "timestamp": 1704067200.0,
                "features": []  # Empty features
            },
            headers=headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestTransactionHistory:
    """Test transaction history endpoint."""
    
    def test_get_empty_history(self, client, auth_token):
        """Test getting transaction history when none exist."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        response = client.get(
            "/api/v1/transactions/history",
            headers=headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert len(data["transactions"]) == 0
    
    def test_get_transaction_by_id(self, client, auth_token):
        """Test retrieving a specific transaction."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # First create a transaction
        create_response = client.post(
            "/api/v1/transactions/predict",
            json={
                "amount": 100.00,
                "timestamp": 1704067200.0,
                "features": [0.1, -0.5, 0.3, 0.2]
            },
            headers=headers
        )
        
        assert create_response.status_code == status.HTTP_200_OK
        transaction_id = create_response.json()["transaction_id"]
        
        # Now retrieve it
        get_response = client.get(
            f"/api/v1/transactions/{transaction_id}",
            headers=headers
        )
        
        assert get_response.status_code == status.HTTP_200_OK
        data = get_response.json()
        assert data["id"] == transaction_id
        assert data["amount"] == 100.00

"""
Fraud alert management routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.database.engine import get_db
from app.models import User, Alert
from app.schemas import AlertResponse

router = APIRouter()


@router.get("/", response_model=list[AlertResponse])
def get_alerts(
    current_user: User = Depends(),
    alert_status: str | None = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get fraud alerts.
    
    - **alert_status**: Filter by status (pending, reviewed, resolved, false_positive)
    - **skip**: Pagination offset
    - **limit**: Maximum results (max 1000)
    """
    
    if limit > 1000:
        limit = 1000
    
    query = db.query(Alert)
    
    if alert_status:
        valid_statuses = ["pending", "reviewed", "resolved", "false_positive"]
        if alert_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid options: {valid_statuses}"
            )
        query = query.filter(Alert.alert_status == alert_status)
    
    alerts = query.order_by(Alert.created_at.desc()).offset(skip).limit(limit).all()
    
    logger.info(f"Retrieved {len(alerts)} alerts for user {current_user.username}")
    
    return alerts


@router.get("/pending/count")
def get_pending_alerts_count(
    current_user: User = Depends(),
    db: Session = Depends(get_db)
):
    """Get count of pending alerts."""
    
    count = db.query(Alert).filter(
        Alert.alert_status == "pending"
    ).count()
    
    return {"pending_count": count}


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: int,
    current_user: User = Depends(),
    db: Session = Depends(get_db)
):
    """Get details of a specific alert."""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    return alert


@router.patch("/{alert_id}/status")
def update_alert_status(
    alert_id: int,
    new_status: str,
    current_user: User = Depends(),
    db: Session = Depends(get_db)
):
    """
    Update alert status.
    
    Valid statuses: pending, reviewed, resolved, false_positive
    """
    
    valid_statuses = ["pending", "reviewed", "resolved", "false_positive"]
    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Valid options: {valid_statuses}"
        )
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    alert.alert_status = new_status
    db.commit()
    db.refresh(alert)
    
    logger.info(f"Alert {alert_id} status updated to {new_status} by {current_user.username}")
    
    return alert

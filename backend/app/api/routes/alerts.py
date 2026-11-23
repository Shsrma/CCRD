"""
Fraud alert management routes.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.database.engine import get_db
from app.models import User, Alert
from app.schemas import AlertResponse
from app.api.dependencies import get_current_user, get_current_user_optional, get_tenant_context, TenantContext

router = APIRouter()


@router.get("", response_model=list[AlertResponse])
@router.get("/", response_model=list[AlertResponse], include_in_schema=False)
def get_alerts(
    current_user: Optional[User] = Depends(get_current_user_optional),
    context: TenantContext = Depends(get_tenant_context),
    alert_status: str | None = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get fraud alerts with tenant isolation.
    """
    if limit > 1000:
        limit = 1000

    query = db.query(Alert)

    # Tenant isolation
    org_id = context.organization.id if context and context.organization else None
    role = context.role if context else "analyst"
    if role != "superadmin" and org_id is not None:
        query = query.filter((Alert.organization_id == org_id) | (Alert.organization_id.is_(None)))

    if alert_status:
        valid_statuses = ["pending", "reviewed", "resolved", "false_positive", "investigating"]
        if alert_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid options: {valid_statuses}"
            )
        query = query.filter(Alert.alert_status == alert_status)

    alerts = query.order_by(Alert.created_at.desc()).offset(skip).limit(limit).all()
    return alerts


@router.put("/{alert_id}")
def update_alert(
    alert_id: int,
    payload: dict = Body(...),
    current_user: Optional[User] = Depends(get_current_user_optional),
    context: TenantContext = Depends(get_tenant_context),
    db: Session = Depends(get_db)
):
    """Update alert status via PUT for dashboard workflow."""
    new_status = payload.get("status")
    if not new_status:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing status")

    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    alert.alert_status = new_status
    db.commit()
    db.refresh(alert)
    return alert

    
    return alerts


@router.get("/pending/count")
def get_pending_alerts_count(
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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

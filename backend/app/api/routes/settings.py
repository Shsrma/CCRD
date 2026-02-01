"""
System settings routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logger import logger
from app.database.engine import get_db
from app.models import User
from app.schemas import GlobalSettings, SettingsUpdate

router = APIRouter()
settings = get_settings()

# In-memory settings (for MVP, use database for production)
system_settings = {
    "fraud_threshold": settings.fraud_threshold,
    "language": "en",
    "timezone": "UTC",
}


@router.get("/", response_model=GlobalSettings)
def get_settings(current_user: User = Depends()):
    """Get current system settings."""
    return system_settings


@router.patch("/", response_model=GlobalSettings)
def update_settings(
    updates: GlobalSettings,
    current_user: User = Depends(),
    db: Session = Depends(get_db)
):
    """
    Update system settings.
    
    Only admin users can modify global settings.
    """
    
    # Check if user has admin role
    if current_user.role not in ["admin", "super_admin"]:
        logger.warning(f"Unauthorized settings update attempt by {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can modify settings"
        )
    
    # Update settings
    if updates.fraud_threshold is not None:
        if not (0.0 <= updates.fraud_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="fraud_threshold must be between 0.0 and 1.0"
            )
        system_settings["fraud_threshold"] = updates.fraud_threshold
        logger.info(f"fraud_threshold updated to {updates.fraud_threshold}")
    
    if updates.language:
        system_settings["language"] = updates.language
        logger.info(f"language updated to {updates.language}")
    
    if updates.timezone:
        system_settings["timezone"] = updates.timezone
        logger.info(f"timezone updated to {updates.timezone}")
    
    return system_settings


@router.patch("/single")
def update_single_setting(
    payload: SettingsUpdate,
    current_user: User = Depends(),
    db: Session = Depends(get_db)
):
    """
    Update a single setting.
    
    Request body:
    {
        "type": "fraud_threshold",
        "value": 0.6
    }
    """
    
    # Check authorization
    if current_user.role not in ["admin", "super_admin"]:
        logger.warning(f"Unauthorized setting update attempt by {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can modify settings"
        )
    
    setting_type = payload.setting_type
    value = payload.value
    
    # Validate and update
    if setting_type == "fraud_threshold":
        try:
            value = float(value)
            if not (0.0 <= value <= 1.0):
                raise ValueError("Must be between 0 and 1")
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="fraud_threshold must be a number between 0 and 1"
            )
        system_settings["fraud_threshold"] = value
    
    elif setting_type == "language":
        system_settings["language"] = str(value)
    
    elif setting_type == "timezone":
        system_settings["timezone"] = str(value)
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown setting: {setting_type}"
        )
    
    logger.info(f"Setting {setting_type} updated to {value} by {current_user.username}")
    
    return {
        "status": "updated",
        "setting": setting_type,
        "value": value,
        "all_settings": system_settings
    }

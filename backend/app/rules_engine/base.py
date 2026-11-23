"""
Base classes for CCRD Rules Engine.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class RuleResult(BaseModel):
    """Result emitted when a deterministic rule triggers."""
    rule_name: str
    reason_code: str
    severity: str = Field(default="MEDIUM", description="LOW, MEDIUM, or HIGH")
    description: str
    risk_boost: float = Field(default=0.0, description="Additive risk score modifier [0.0, 1.0]")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseRule(ABC):
    """Abstract base class for all fraud rules."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the rule."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable explanation of rule logic."""
        pass

    @abstractmethod
    def evaluate(self, transaction: Any) -> Optional[RuleResult]:
        """
        Evaluate rule against transaction.
        Returns RuleResult if triggered, otherwise None.
        """
        pass

"""
Rules Engine coordinator for CCRD Fraud Intelligence Platform.
Executes configured rules in sequence, aggregates triggered results, and produces reason codes.
"""

from typing import List, Dict, Any, Tuple
from backend.app.rules_engine.base import BaseRule, RuleResult
from backend.app.rules_engine.rules import (
    HighAmountRule,
    HighVelocityRule,
    UnusualNightHourRule,
    MultipleFailedAttemptsRule,
    UntrustedDeviceRule,
    NewAccountHighRiskCategoryRule,
)


class RulesEngine:
    """
    Evaluates transactions across a registered collection of business fraud rules.
    """

    def __init__(self, rules: List[BaseRule] = None):
        if rules is not None:
            self.rules = rules
        else:
            # Default production rule suite
            self.rules = [
                HighAmountRule(absolute_threshold=5000.0, ratio_threshold=8.0),
                HighVelocityRule(velocity_threshold=5),
                UnusualNightHourRule(amount_threshold=300.0),
                MultipleFailedAttemptsRule(failure_threshold=3),
                UntrustedDeviceRule(trust_threshold=0.25),
                NewAccountHighRiskCategoryRule(max_age_days=14, min_amount=200.0),
            ]

    def evaluate(self, transaction: Any) -> Tuple[List[RuleResult], float]:
        """
        Executes all active rules against a transaction.

        Returns:
            Tuple of (triggered_rule_results, composite_rule_risk_modifier)
        """
        triggered = []
        total_boost = 0.0

        for rule in self.rules:
            result = rule.evaluate(transaction)
            if result is not None:
                triggered.append(result)
                total_boost += result.risk_boost

        # Cap the rule risk modifier to at most 0.85
        rule_modifier = min(total_boost, 0.85)
        return triggered, rule_modifier

    def get_registered_rules(self) -> List[Dict[str, Any]]:
        """Returns metadata about all registered rules."""
        return [
            {
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
            }
            for rule in self.rules
        ]

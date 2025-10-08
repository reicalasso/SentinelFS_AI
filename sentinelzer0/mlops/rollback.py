"""
Automated Rollback System

Provides automated rollback mechanisms with health checks, performance monitoring,
and intelligent decision-making for model deployments.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

from .version_manager import ModelVersionManager
from .model_registry import ModelRegistry, ModelStage


class RollbackStrategy(Enum):
    """Rollback strategy types."""
    IMMEDIATE = "immediate"  # Rollback immediately on failure
    GRADUAL = "gradual"  # Gradually shift traffic back
    MANUAL = "manual"  # Require manual approval


@dataclass
class HealthCheck:
    """Health check result."""
    timestamp: str
    version: str
    is_healthy: bool
    error_rate: float
    avg_latency_ms: float
    success_rate: float
    details: Dict[str, Any]


@dataclass
class RollbackEvent:
    """Rollback event record."""
    event_id: str
    timestamp: str
    from_version: str
    to_version: str
    reason: str
    strategy: str
    triggered_by: str = "system"
    success: bool = False
    details: Dict[str, Any] = None


class RollbackManager:
    """
    Manages automated rollback with health monitoring.
    
    Features:
    - Continuous health monitoring
    - Automatic rollback on failures
    - Configurable rollback strategies
    - Rollback history and audit trail
    """
    
    def __init__(
        self,
        version_manager: ModelVersionManager,
        registry: ModelRegistry,
        rollback_dir: str = "models/rollback"
    ):
        """Initialize rollback manager."""
        self.version_manager = version_manager
        self.registry = registry
        self.rollback_dir = Path(rollback_dir)
        self.rollback_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.error_rate_threshold = 0.10  # 10% error rate
        self.latency_threshold_ms = 100.0  # 100ms latency
        self.min_requests_for_check = 50  # Minimum requests before health check
        
        # Rollback history
        self.rollback_history: List[RollbackEvent] = []
        self._load_history()
    
    def check_health(self, version: str, metrics: Dict[str, Any]) -> HealthCheck:
        """
        Check health of a deployed model.
        
        Args:
            version: Model version
            metrics: Current metrics
        
        Returns:
            HealthCheck result
        """
        error_rate = metrics.get('error_rate', 0.0)
        avg_latency = metrics.get('avg_latency_ms', 0.0)
        total_requests = metrics.get('total_requests', 0)
        failed_requests = metrics.get('failed_requests', 0)
        
        success_rate = 1.0 - error_rate if total_requests > 0 else 1.0
        
        # Determine if healthy
        is_healthy = (
            error_rate < self.error_rate_threshold and
            avg_latency < self.latency_threshold_ms and
            total_requests >= self.min_requests_for_check
        )
        
        health_check = HealthCheck(
            timestamp=datetime.now().isoformat(),
            version=version,
            is_healthy=is_healthy,
            error_rate=error_rate,
            avg_latency_ms=avg_latency,
            success_rate=success_rate,
            details={
                'total_requests': total_requests,
                'failed_requests': failed_requests,
                'thresholds': {
                    'error_rate': self.error_rate_threshold,
                    'latency_ms': self.latency_threshold_ms
                }
            }
        )
        
        # Update registry health status
        status = "healthy" if is_healthy else "unhealthy"
        try:
            self.registry.update_health_status(version, status, metrics)
        except ValueError:
            pass  # Version might not be registered yet
        
        if not is_healthy:
            self.logger.warning(
                f"Health check failed for {version}: "
                f"error_rate={error_rate:.2%}, latency={avg_latency:.1f}ms"
            )
        
        return health_check
    
    def should_rollback(
        self,
        current_version: str,
        health_check: HealthCheck,
        strategy: RollbackStrategy = RollbackStrategy.IMMEDIATE
    ) -> bool:
        """
        Determine if rollback should be triggered.
        
        Args:
            current_version: Current production version
            health_check: Latest health check
            strategy: Rollback strategy
        
        Returns:
            True if rollback should be triggered
        """
        if strategy == RollbackStrategy.MANUAL:
            return False  # Manual rollback only
        
        if not health_check.is_healthy:
            if strategy == RollbackStrategy.IMMEDIATE:
                return True
            elif strategy == RollbackStrategy.GRADUAL:
                # Check if unhealthy for sustained period
                return self._is_sustained_unhealthy(current_version)
        
        return False
    
    def execute_rollback(
        self,
        from_version: str,
        to_version: Optional[str] = None,
        reason: str = "Automated rollback",
        strategy: RollbackStrategy = RollbackStrategy.IMMEDIATE
    ) -> RollbackEvent:
        """
        Execute a rollback.
        
        Args:
            from_version: Version to rollback from
            to_version: Version to rollback to (previous production if None)
            reason: Rollback reason
            strategy: Rollback strategy
        
        Returns:
            RollbackEvent
        """
        # Determine target version if not specified
        if to_version is None:
            to_version = self._get_previous_production_version(from_version)
            if not to_version:
                raise ValueError("No previous production version found for rollback")
        
        # Verify target version exists
        target_model = self.version_manager.get_version(to_version)
        if not target_model:
            raise ValueError(f"Target version not found: {to_version}")
        
        # Create rollback event
        event = RollbackEvent(
            event_id=f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            from_version=from_version,
            to_version=to_version,
            reason=reason,
            strategy=strategy.value,
            details={
                'target_model_path': str(target_model.model_path),
                'rollback_method': 'automated' if strategy != RollbackStrategy.MANUAL else 'manual'
            }
        )
        
        try:
            # Execute rollback in registry
            # Demote current version
            self.registry.registry[from_version].stage = ModelStage.STAGING.value
            
            # Promote target version
            self.registry.registry[to_version].stage = ModelStage.PRODUCTION.value
            self.registry.registry[to_version].promoted_at = datetime.now().isoformat()
            self.registry.registry[to_version].promoted_by = "rollback_system"
            
            # Save registry changes
            self.registry._save_registry()
            
            event.success = True
            self.logger.info(
                f"Rollback successful: {from_version} -> {to_version}"
            )
        
        except Exception as e:
            event.success = False
            event.details['error'] = str(e)
            self.logger.error(f"Rollback failed: {e}")
        
        # Record event
        self.rollback_history.append(event)
        self._save_history()
        
        return event
    
    def get_rollback_history(
        self,
        limit: int = 10
    ) -> List[RollbackEvent]:
        """Get recent rollback history."""
        return sorted(
            self.rollback_history,
            key=lambda e: e.timestamp,
            reverse=True
        )[:limit]
    
    def _get_previous_production_version(self, current_version: str) -> Optional[str]:
        """Get the previous production version before current."""
        # Get all production versions from registry
        all_entries = list(self.registry.registry.values())
        
        # Filter production versions
        prod_entries = [
            e for e in all_entries
            if e.version != current_version and
            e.stage == ModelStage.PRODUCTION.value or
            e.promoted_at is not None
        ]
        
        if not prod_entries:
            return None
        
        # Sort by promotion time
        prod_entries.sort(
            key=lambda e: e.promoted_at or e.registered_at,
            reverse=True
        )
        
        return prod_entries[0].version if prod_entries else None
    
    def _is_sustained_unhealthy(
        self,
        version: str,
        duration_minutes: int = 5
    ) -> bool:
        """Check if version has been unhealthy for sustained period."""
        # Get registry entry
        entry = self.registry.registry.get(version)
        if not entry:
            return False
        
        if entry.health_status != "unhealthy":
            return False
        
        # Check if last health check was recent
        if entry.last_health_check:
            last_check = datetime.fromisoformat(entry.last_health_check)
            now = datetime.now()
            
            if (now - last_check).total_seconds() > duration_minutes * 60:
                return True
        
        return False
    
    def _save_history(self):
        """Save rollback history."""
        history_path = self.rollback_dir / "rollback_history.json"
        history_data = [asdict(event) for event in self.rollback_history]
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def _load_history(self):
        """Load rollback history."""
        history_path = self.rollback_dir / "rollback_history.json"
        if not history_path.exists():
            return
        
        try:
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            
            self.rollback_history = [
                RollbackEvent(**data) for data in history_data
            ]
        except Exception as e:
            self.logger.error(f"Error loading rollback history: {e}")

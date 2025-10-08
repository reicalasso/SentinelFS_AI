"""
Model Registry with Approval Workflows

Provides model registry with approval workflows, staging management,
and production promotion capabilities.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

from .version_manager import ModelVersionManager, ModelVersion, VersionStatus


class ModelStage(Enum):
    """Model lifecycle stages."""
    NONE = "none"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ApprovalStatus(Enum):
    """Approval status for stage transitions."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ApprovalRequest:
    """Request for model stage promotion."""
    request_id: str
    version: str
    from_stage: str
    to_stage: str
    requested_by: str
    requested_at: str
    status: str = ApprovalStatus.PENDING.value
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApprovalRequest':
        return cls(**data)


@dataclass
class RegistryEntry:
    """Entry in the model registry."""
    version: str
    stage: str
    registered_at: str
    registered_by: str
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None
    deployment_info: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    health_status: str = "healthy"
    last_health_check: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegistryEntry':
        return cls(**data)


class ModelRegistry:
    """
    Model registry with approval workflows and stage management.
    
    Features:
    - Model registration and lifecycle management
    - Approval workflows for stage transitions
    - Staging and production environment management
    - Health checks and performance tracking
    - Audit trail for all operations
    """
    
    def __init__(
        self,
        version_manager: ModelVersionManager,
        registry_dir: str = "models/registry"
    ):
        """
        Initialize model registry.
        
        Args:
            version_manager: ModelVersionManager instance
            registry_dir: Directory for registry data
        """
        self.version_manager = version_manager
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Registry data
        self.registry: Dict[str, RegistryEntry] = {}
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        
        # Load existing data
        self._load_registry()
        self._load_approval_requests()
    
    def register_model(
        self,
        version: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        registered_by: str = "system",
        tags: Optional[List[str]] = None
    ) -> RegistryEntry:
        """
        Register a model version in the registry.
        
        Args:
            version: Model version
            stage: Initial stage
            registered_by: User registering the model
            tags: Tags for the model
        
        Returns:
            Created RegistryEntry
        """
        # Verify version exists
        model_version = self.version_manager.get_version(version)
        if not model_version:
            raise ValueError(f"Version not found: {version}")
        
        # Check if already registered
        if version in self.registry:
            raise ValueError(f"Version already registered: {version}")
        
        # Create registry entry
        entry = RegistryEntry(
            version=version,
            stage=stage.value,
            registered_at=datetime.now().isoformat(),
            registered_by=registered_by,
            tags=tags or []
        )
        
        self.registry[version] = entry
        self._save_registry()
        
        self.logger.info(f"Registered model version {version} in stage {stage.value}")
        return entry
    
    def request_promotion(
        self,
        version: str,
        to_stage: ModelStage,
        requested_by: str,
        notes: str = ""
    ) -> ApprovalRequest:
        """
        Request promotion of a model to a new stage.
        
        Args:
            version: Version to promote
            to_stage: Target stage
            requested_by: User requesting promotion
            notes: Additional notes
        
        Returns:
            Created ApprovalRequest
        """
        # Verify version is registered
        entry = self.registry.get(version)
        if not entry:
            raise ValueError(f"Version not registered: {version}")
        
        from_stage = ModelStage(entry.stage)
        
        # Validate stage transition
        self._validate_stage_transition(from_stage, to_stage)
        
        # Generate request ID
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{version}"
        
        # Create approval request
        request = ApprovalRequest(
            request_id=request_id,
            version=version,
            from_stage=from_stage.value,
            to_stage=to_stage.value,
            requested_by=requested_by,
            requested_at=datetime.now().isoformat(),
            notes=notes
        )
        
        self.approval_requests[request_id] = request
        self._save_approval_requests()
        
        self.logger.info(
            f"Created promotion request {request_id}: "
            f"{version} from {from_stage.value} to {to_stage.value}"
        )
        
        return request
    
    def approve_promotion(
        self,
        request_id: str,
        approved_by: str,
        notes: str = ""
    ) -> RegistryEntry:
        """
        Approve a promotion request.
        
        Args:
            request_id: Request ID
            approved_by: User approving the request
            notes: Approval notes
        
        Returns:
            Updated RegistryEntry
        """
        request = self.approval_requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")
        
        if request.status != ApprovalStatus.PENDING.value:
            raise ValueError(f"Request already processed: {request.status}")
        
        # Update request status
        request.status = ApprovalStatus.APPROVED.value
        request.approved_by = approved_by
        request.approved_at = datetime.now().isoformat()
        if notes:
            request.notes = f"{request.notes}\nApproval notes: {notes}"
        
        # Promote model
        entry = self.registry[request.version]
        entry.stage = request.to_stage
        entry.promoted_at = datetime.now().isoformat()
        entry.promoted_by = approved_by
        
        # Update version manager status
        version_status = self._stage_to_version_status(ModelStage(request.to_stage))
        self.version_manager.promote_version(request.version, version_status)
        
        # Save changes
        self._save_registry()
        self._save_approval_requests()
        
        self.logger.info(f"Approved promotion request {request_id}")
        return entry
    
    def reject_promotion(
        self,
        request_id: str,
        rejected_by: str,
        reason: str
    ):
        """
        Reject a promotion request.
        
        Args:
            request_id: Request ID
            rejected_by: User rejecting the request
            reason: Rejection reason
        """
        request = self.approval_requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")
        
        if request.status != ApprovalStatus.PENDING.value:
            raise ValueError(f"Request already processed: {request.status}")
        
        # Update request status
        request.status = ApprovalStatus.REJECTED.value
        request.approved_by = rejected_by
        request.approved_at = datetime.now().isoformat()
        request.rejection_reason = reason
        
        self._save_approval_requests()
        
        self.logger.info(f"Rejected promotion request {request_id}: {reason}")
    
    def get_models_by_stage(self, stage: ModelStage) -> List[RegistryEntry]:
        """Get all models in a specific stage."""
        return [
            entry for entry in self.registry.values()
            if entry.stage == stage.value
        ]
    
    def get_production_model(self) -> Optional[RegistryEntry]:
        """Get the current production model."""
        production_models = self.get_models_by_stage(ModelStage.PRODUCTION)
        if not production_models:
            return None
        
        # Return most recently promoted
        production_models.sort(
            key=lambda e: e.promoted_at or e.registered_at,
            reverse=True
        )
        return production_models[0]
    
    def update_health_status(
        self,
        version: str,
        status: str = "healthy",
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Update health status and performance metrics for a model.
        
        Args:
            version: Model version
            status: Health status
            metrics: Performance metrics
        """
        entry = self.registry.get(version)
        if not entry:
            raise ValueError(f"Version not registered: {version}")
        
        entry.health_status = status
        entry.last_health_check = datetime.now().isoformat()
        
        if metrics:
            entry.performance_metrics.update(metrics)
        
        self._save_registry()
        
        self.logger.info(f"Updated health status for {version}: {status}")
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [
            req for req in self.approval_requests.values()
            if req.status == ApprovalStatus.PENDING.value
        ]
    
    def get_audit_trail(self, version: str) -> Dict[str, Any]:
        """
        Get complete audit trail for a model version.
        
        Args:
            version: Model version
        
        Returns:
            Audit trail information
        """
        entry = self.registry.get(version)
        if not entry:
            raise ValueError(f"Version not registered: {version}")
        
        # Get all approval requests for this version
        requests = [
            req.to_dict() for req in self.approval_requests.values()
            if req.version == version
        ]
        
        # Get version metadata
        model_version = self.version_manager.get_version(version)
        
        audit_trail = {
            "version": version,
            "registry_entry": entry.to_dict(),
            "version_metadata": model_version.metadata.to_dict() if model_version else {},
            "approval_requests": requests,
            "stage_history": self._get_stage_history(version)
        }
        
        return audit_trail
    
    def _validate_stage_transition(
        self,
        from_stage: ModelStage,
        to_stage: ModelStage
    ):
        """Validate if a stage transition is allowed."""
        # Define valid transitions
        valid_transitions = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: [ModelStage.DEVELOPMENT],  # Can restore from archive
        }
        
        allowed = valid_transitions.get(from_stage, [])
        if to_stage not in allowed:
            raise ValueError(
                f"Invalid stage transition: {from_stage.value} -> {to_stage.value}"
            )
    
    def _stage_to_version_status(self, stage: ModelStage) -> VersionStatus:
        """Convert ModelStage to VersionStatus."""
        mapping = {
            ModelStage.DEVELOPMENT: VersionStatus.DEVELOPMENT,
            ModelStage.STAGING: VersionStatus.STAGING,
            ModelStage.PRODUCTION: VersionStatus.PRODUCTION,
            ModelStage.ARCHIVED: VersionStatus.ARCHIVED,
        }
        return mapping[stage]
    
    def _get_stage_history(self, version: str) -> List[Dict[str, Any]]:
        """Get stage history for a version."""
        history = []
        
        # Add registration event
        entry = self.registry.get(version)
        if entry:
            history.append({
                "event": "registered",
                "stage": entry.stage,
                "timestamp": entry.registered_at,
                "user": entry.registered_by
            })
            
            # Add promotion event if exists
            if entry.promoted_at:
                history.append({
                    "event": "promoted",
                    "stage": entry.stage,
                    "timestamp": entry.promoted_at,
                    "user": entry.promoted_by
                })
        
        return history
    
    def _save_registry(self):
        """Save registry to file."""
        registry_path = self.registry_dir / "registry.json"
        registry_data = {
            version: entry.to_dict()
            for version, entry in self.registry.items()
        }
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def _load_registry(self):
        """Load registry from file."""
        registry_path = self.registry_dir / "registry.json"
        if not registry_path.exists():
            return
        
        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            self.registry = {
                version: RegistryEntry.from_dict(data)
                for version, data in registry_data.items()
            }
        except Exception as e:
            self.logger.error(f"Error loading registry: {e}")
    
    def _save_approval_requests(self):
        """Save approval requests to file."""
        requests_path = self.registry_dir / "approval_requests.json"
        requests_data = {
            req_id: req.to_dict()
            for req_id, req in self.approval_requests.items()
        }
        with open(requests_path, 'w') as f:
            json.dump(requests_data, f, indent=2)
    
    def _load_approval_requests(self):
        """Load approval requests from file."""
        requests_path = self.registry_dir / "approval_requests.json"
        if not requests_path.exists():
            return
        
        try:
            with open(requests_path, 'r') as f:
                requests_data = json.load(f)
            
            self.approval_requests = {
                req_id: ApprovalRequest.from_dict(data)
                for req_id, data in requests_data.items()
            }
        except Exception as e:
            self.logger.error(f"Error loading approval requests: {e}")

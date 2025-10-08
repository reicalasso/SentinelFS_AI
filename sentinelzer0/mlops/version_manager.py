"""
Model Versioning System

Provides comprehensive model version management with metadata tracking,
version comparison, and lifecycle management.
"""

import json
import hashlib
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import logging


class VersionStatus(Enum):
    """Model version status."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    """Metadata for a model version."""
    version: str
    created_at: str
    created_by: str
    framework: str = "pytorch"
    framework_version: str = "2.0.0"
    model_type: str = "hybrid_detector"
    architecture: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_dataset: str = ""
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    file_hash: str = ""
    file_size: int = 0
    status: str = VersionStatus.DEVELOPMENT.value
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelVersion:
    """Complete model version with metadata and file path."""
    metadata: ModelMetadata
    model_path: Path
    
    @property
    def version(self) -> str:
        return self.metadata.version
    
    @property
    def status(self) -> VersionStatus:
        return VersionStatus(self.metadata.status)
    
    def set_status(self, status: VersionStatus):
        """Update version status."""
        self.metadata.status = status.value


class ModelVersionManager:
    """
    Manages model versions with metadata tracking and lifecycle management.
    
    Features:
    - Version creation and registration
    - Metadata management
    - Version comparison
    - File hash verification
    - Lifecycle management (dev -> staging -> production)
    """
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize version manager.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / "versions"
        self.metadata_dir = self.base_dir / "metadata"
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing versions
        self.versions: Dict[str, ModelVersion] = {}
        self._load_versions()
    
    def create_version(
        self,
        model_path: str,
        version: Optional[str] = None,
        created_by: str = "system",
        training_metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        notes: str = "",
        tags: Optional[List[str]] = None,
        parent_version: Optional[str] = None
    ) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            model_path: Path to model file
            version: Version string (auto-generated if None)
            created_by: Creator identifier
            training_metrics: Training metrics
            hyperparameters: Model hyperparameters
            notes: Version notes
            tags: Version tags
            parent_version: Parent version (for incremental updates)
        
        Returns:
            Created ModelVersion
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version()
        
        # Check if version already exists
        if version in self.versions:
            raise ValueError(f"Version {version} already exists")
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(model_path)
        file_size = model_path.stat().st_size
        
        # Create metadata
        metadata = ModelMetadata(
            version=version,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            training_metrics=training_metrics or {},
            hyperparameters=hyperparameters or {},
            file_hash=file_hash,
            file_size=file_size,
            notes=notes,
            tags=tags or [],
            parent_version=parent_version
        )
        
        # Copy model file to versions directory
        version_path = self.versions_dir / f"model_{version}.pt"
        shutil.copy2(model_path, version_path)
        
        # Create ModelVersion
        model_version = ModelVersion(metadata=metadata, model_path=version_path)
        
        # Save metadata
        self._save_metadata(metadata)
        
        # Add to versions
        self.versions[version] = model_version
        
        self.logger.info(f"Created model version: {version}")
        return model_version
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get a specific version."""
        return self.versions.get(version)
    
    def list_versions(
        self,
        status: Optional[VersionStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelVersion]:
        """
        List versions with optional filtering.
        
        Args:
            status: Filter by status
            tags: Filter by tags
        
        Returns:
            List of matching versions
        """
        versions = list(self.versions.values())
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        if tags:
            versions = [
                v for v in versions
                if any(tag in v.metadata.tags for tag in tags)
            ]
        
        # Sort by creation date (newest first)
        versions.sort(
            key=lambda v: v.metadata.created_at,
            reverse=True
        )
        
        return versions
    
    def get_latest_version(
        self,
        status: Optional[VersionStatus] = None
    ) -> Optional[ModelVersion]:
        """Get the latest version, optionally filtered by status."""
        versions = self.list_versions(status=status)
        return versions[0] if versions else None
    
    def get_production_version(self) -> Optional[ModelVersion]:
        """Get the current production version."""
        return self.get_latest_version(status=VersionStatus.PRODUCTION)
    
    def promote_version(
        self,
        version: str,
        target_status: VersionStatus
    ) -> ModelVersion:
        """
        Promote a version to a new status.
        
        Args:
            version: Version to promote
            target_status: Target status
        
        Returns:
            Updated ModelVersion
        """
        model_version = self.get_version(version)
        if not model_version:
            raise ValueError(f"Version not found: {version}")
        
        old_status = model_version.status
        model_version.set_status(target_status)
        
        # Save updated metadata
        self._save_metadata(model_version.metadata)
        
        self.logger.info(
            f"Promoted version {version}: {old_status.value} -> {target_status.value}"
        )
        
        return model_version
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            version1: First version
            version2: Second version
        
        Returns:
            Comparison results
        """
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": self._compare_metrics(
                v1.metadata.training_metrics,
                v2.metadata.training_metrics
            ),
            "hyperparameters_diff": self._compare_dicts(
                v1.metadata.hyperparameters,
                v2.metadata.hyperparameters
            ),
            "file_size_diff": v2.metadata.file_size - v1.metadata.file_size,
            "created_at_diff": (
                datetime.fromisoformat(v2.metadata.created_at) -
                datetime.fromisoformat(v1.metadata.created_at)
            ).total_seconds()
        }
        
        return comparison
    
    def verify_integrity(self, version: str) -> bool:
        """
        Verify the integrity of a model file.
        
        Args:
            version: Version to verify
        
        Returns:
            True if integrity check passes
        """
        model_version = self.get_version(version)
        if not model_version:
            raise ValueError(f"Version not found: {version}")
        
        if not model_version.model_path.exists():
            self.logger.error(f"Model file missing for version {version}")
            return False
        
        current_hash = self._calculate_file_hash(model_version.model_path)
        expected_hash = model_version.metadata.file_hash
        
        if current_hash != expected_hash:
            self.logger.error(
                f"Hash mismatch for version {version}: "
                f"expected {expected_hash}, got {current_hash}"
            )
            return False
        
        return True
    
    def delete_version(self, version: str, force: bool = False):
        """
        Delete a version.
        
        Args:
            version: Version to delete
            force: Force deletion even if in production
        """
        model_version = self.get_version(version)
        if not model_version:
            raise ValueError(f"Version not found: {version}")
        
        # Prevent deleting production versions without force
        if model_version.status == VersionStatus.PRODUCTION and not force:
            raise ValueError(
                f"Cannot delete production version {version} without force=True"
            )
        
        # Delete model file
        if model_version.model_path.exists():
            model_version.model_path.unlink()
        
        # Delete metadata file
        metadata_path = self.metadata_dir / f"metadata_{version}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from versions dict
        del self.versions[version]
        
        self.logger.info(f"Deleted version: {version}")
    
    def _generate_version(self) -> str:
        """Generate a new version string."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to file."""
        metadata_path = self.metadata_dir / f"metadata_{metadata.version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _load_versions(self):
        """Load all existing versions."""
        for metadata_path in self.metadata_dir.glob("metadata_*.json"):
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata = ModelMetadata.from_dict(metadata_dict)
                version = metadata.version
                
                # Find model file
                model_path = self.versions_dir / f"model_{version}.pt"
                if not model_path.exists():
                    self.logger.warning(
                        f"Model file missing for version {version}"
                    )
                    continue
                
                # Create ModelVersion
                model_version = ModelVersion(
                    metadata=metadata,
                    model_path=model_path
                )
                
                self.versions[version] = model_version
            
            except Exception as e:
                self.logger.error(f"Error loading metadata from {metadata_path}: {e}")
    
    def _compare_metrics(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare two metrics dictionaries."""
        all_keys = set(metrics1.keys()) | set(metrics2.keys())
        comparison = {}
        
        for key in all_keys:
            v1 = metrics1.get(key, 0.0)
            v2 = metrics2.get(key, 0.0)
            comparison[key] = {
                "version1": v1,
                "version2": v2,
                "diff": v2 - v1,
                "percent_change": ((v2 - v1) / v1 * 100) if v1 != 0 else 0.0
            }
        
        return comparison
    
    def _compare_dicts(
        self,
        dict1: Dict[str, Any],
        dict2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two dictionaries."""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        diff = {}
        
        for key in all_keys:
            v1 = dict1.get(key)
            v2 = dict2.get(key)
            
            if v1 != v2:
                diff[key] = {"version1": v1, "version2": v2}
        
        return diff

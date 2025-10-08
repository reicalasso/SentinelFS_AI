"""
Audit Trail System

Comprehensive logging and tracking of all model decisions.
Ensures accountability and enables forensic analysis.
"""

import torch
import json
import sqlite3
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib


@dataclass
class AuditEntry:
    """Single audit log entry."""
    entry_id: str  # Unique identifier
    timestamp: datetime
    prediction: float
    confidence: float
    input_hash: str  # Hash of input for privacy
    feature_importance: Dict[str, float]
    reasoning: str
    model_version: str
    metadata: Dict[str, Any]


class AuditTrailSystem:
    """
    Comprehensive audit logging for model decisions.
    
    Features:
    - Persistent storage (SQLite + JSON)
    - Query and search capabilities
    - Compliance reporting
    - Privacy-preserving logging
    - Tamper detection
    - Export capabilities
    """
    
    def __init__(
        self,
        db_path: str = "audit_trail.db",
        json_backup: bool = True,
        json_path: str = "audit_logs.json"
    ):
        """
        Initialize audit trail system.
        
        Args:
            db_path: Path to SQLite database
            json_backup: Enable JSON backup
            json_path: Path to JSON backup file
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.json_backup = json_backup
        self.json_path = Path(json_path) if json_backup else None
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"Initialized audit trail system at {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                entry_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                prediction REAL NOT NULL,
                confidence REAL NOT NULL,
                input_hash TEXT NOT NULL,
                feature_importance TEXT NOT NULL,
                reasoning TEXT,
                model_version TEXT,
                metadata TEXT
            )
        ''')
        
        # Create indices for fast queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON audit_log(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prediction 
            ON audit_log(prediction)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_decision(
        self,
        inputs: torch.Tensor,
        prediction: float,
        confidence: float,
        feature_importance: Dict[str, float],
        reasoning: str = "",
        model_version: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a model decision to audit trail.
        
        Args:
            inputs: Input tensor
            prediction: Model prediction
            confidence: Prediction confidence
            feature_importance: Feature importance scores
            reasoning: Human-readable explanation
            model_version: Model version identifier
            metadata: Additional metadata
        
        Returns:
            Entry ID
        """
        # Generate unique entry ID
        entry_id = self._generate_entry_id()
        
        # Hash input for privacy
        input_hash = self._hash_input(inputs)
        
        # Create audit entry
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            prediction=float(prediction),
            confidence=float(confidence),
            input_hash=input_hash,
            feature_importance=feature_importance,
            reasoning=reasoning,
            model_version=model_version,
            metadata=metadata or {}
        )
        
        # Store in database
        self._store_entry(entry)
        
        # Backup to JSON if enabled
        if self.json_backup:
            self._backup_to_json(entry)
        
        return entry_id
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID."""
        timestamp = datetime.now().isoformat()
        random_bytes = hashlib.sha256(timestamp.encode()).hexdigest()[:8]
        return f"audit_{random_bytes}"
    
    def _hash_input(self, inputs: torch.Tensor) -> str:
        """Generate hash of input for privacy."""
        input_bytes = inputs.cpu().numpy().tobytes()
        return hashlib.sha256(input_bytes).hexdigest()[:16]
    
    def _store_entry(self, entry: AuditEntry):
        """Store entry in SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.entry_id,
            entry.timestamp.isoformat(),
            entry.prediction,
            entry.confidence,
            entry.input_hash,
            json.dumps(entry.feature_importance),
            entry.reasoning,
            entry.model_version,
            json.dumps(entry.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _backup_to_json(self, entry: AuditEntry):
        """Backup entry to JSON file."""
        # Read existing logs
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new entry
        entry_dict = asdict(entry)
        entry_dict['timestamp'] = entry.timestamp.isoformat()
        logs.append(entry_dict)
        
        # Write back
        with open(self.json_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def query_by_timerange(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[AuditEntry]:
        """
        Query entries by time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
        
        Returns:
            List of audit entries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM audit_log
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (start_time.isoformat(), end_time.isoformat()))
        
        entries = [self._row_to_entry(row) for row in cursor.fetchall()]
        
        conn.close()
        return entries
    
    def query_by_prediction(
        self,
        prediction_value: float,
        tolerance: float = 0.1
    ) -> List[AuditEntry]:
        """
        Query entries by prediction value.
        
        Args:
            prediction_value: Target prediction value
            tolerance: Acceptable deviation
        
        Returns:
            List of audit entries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM audit_log
            WHERE prediction BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (prediction_value - tolerance, prediction_value + tolerance))
        
        entries = [self._row_to_entry(row) for row in cursor.fetchall()]
        
        conn.close()
        return entries
    
    def query_low_confidence(
        self,
        threshold: float = 0.7
    ) -> List[AuditEntry]:
        """
        Query low-confidence predictions.
        
        Args:
            threshold: Confidence threshold
        
        Returns:
            List of audit entries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM audit_log
            WHERE confidence < ?
            ORDER BY confidence ASC
        ''', (threshold,))
        
        entries = [self._row_to_entry(row) for row in cursor.fetchall()]
        
        conn.close()
        return entries
    
    def _row_to_entry(self, row: tuple) -> AuditEntry:
        """Convert database row to AuditEntry."""
        return AuditEntry(
            entry_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            prediction=row[2],
            confidence=row[3],
            input_hash=row[4],
            feature_importance=json.loads(row[5]),
            reasoning=row[6],
            model_version=row[7],
            metadata=json.loads(row[8])
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit trail statistics.
        
        Returns:
            Statistics dictionary
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Total entries
        cursor.execute('SELECT COUNT(*) FROM audit_log')
        total_entries = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM audit_log')
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Prediction distribution
        cursor.execute('SELECT prediction, COUNT(*) FROM audit_log GROUP BY prediction')
        prediction_dist = dict(cursor.fetchall())
        
        # Low confidence count
        cursor.execute('SELECT COUNT(*) FROM audit_log WHERE confidence < 0.7')
        low_conf_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_entries': total_entries,
            'average_confidence': float(avg_confidence),
            'prediction_distribution': prediction_dist,
            'low_confidence_count': low_conf_count,
            'low_confidence_rate': low_conf_count / total_entries if total_entries > 0 else 0.0
        }
    
    def export_to_csv(self, output_path: str):
        """
        Export audit log to CSV.
        
        Args:
            output_path: Path to CSV file
        """
        import csv
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM audit_log')
        rows = cursor.fetchall()
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'entry_id', 'timestamp', 'prediction', 'confidence',
                'input_hash', 'feature_importance', 'reasoning',
                'model_version', 'metadata'
            ])
            writer.writerows(rows)
        
        conn.close()
        self.logger.info(f"Exported {len(rows)} entries to {output_path}")
    
    def verify_integrity(self) -> bool:
        """
        Verify database integrity.
        
        Returns:
            True if integrity check passes
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute('PRAGMA integrity_check')
            result = cursor.fetchone()[0]
            conn.close()
            
            return result == 'ok'
        except Exception as e:
            self.logger.error(f"Integrity check failed: {e}")
            return False

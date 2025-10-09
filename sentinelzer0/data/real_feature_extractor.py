"""
Real-world feature extraction from actual file system operations.
This module extracts meaningful features from live file system events for threat detection.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import hashlib
import math

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RealFeatureExtractor:
    """
    Extract security-relevant features from real file system operations.
    
    Features extracted:
    1. Temporal features: access patterns, time-based anomalies
    2. File features: size, type, entropy, modification patterns
    3. User behavior: access velocity, pattern deviation
    4. System features: resource usage, concurrent operations
    5. Security indicators: suspicious patterns, rapid changes
    """
    
    # File extensions categorized by risk and type
    EXECUTABLE_EXTENSIONS = {'.exe', '.dll', '.so', '.dylib', '.bin', '.bat', '.sh', '.cmd', '.msi', '.app'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.txt', '.csv'}
    COMPRESSED_EXTENSIONS = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico'}
    CODE_EXTENSIONS = {'.py', '.js', '.java', '.c', '.cpp', '.h', '.rs', '.go', '.rb'}
    ENCRYPTED_EXTENSIONS = {'.enc', '.encrypted', '.locked', '.crypted', '.crypt', '.aes', '.rsa', '.cerber', '.locky', '.wannacry'}
    
    # Suspicious file patterns (ransomware indicators)
    RANSOMWARE_PATTERNS = {
        'extensions': {
            '.locked', '.encrypted', '.crypted', '.enc', '.crypt', '.crypto', 
            '.aes', '.rsa', '.cerber', '.locky', '.wannacry', '.zepto', '.osiris',
            '.zzzzz', '.dharma', '.wallet', '.wcry', '.WNCRY', '.onion', '.exx',
            '.ezz', '.ecc', '.ezz', '.abc', '.xyz', '.zzz', '.microc', '.dll5'
        },
        'filenames': {
            'readme.txt', 'how_to_decrypt.txt', 'decrypt_instructions.txt',
            'help_decrypt.txt', 'recovery_file.txt', 'restore_files.txt',
            'help_recover_instructions.txt', 'readme_now.txt', 'read_me.txt'
        }
    }
    
    # Suspicious system paths
    CRITICAL_SYSTEM_PATHS = {
        '/etc/passwd', '/etc/shadow', '/etc/sudoers', '/etc/hosts',
        '/boot/', '/sys/', '/proc/', 'C:\\Windows\\System32',
        'C:\\Windows\\SysWOW64', '/usr/bin/', '/usr/sbin/', '/var/log/'
    }
    
    # Trusted process names (allowlist - known legitimate applications)
    TRUSTED_PROCESSES = {
        # Document viewers/editors
        'evince', 'okular', 'libreoffice', 'soffice', 'abiword', 'gedit', 'kate', 
        'nano', 'vim', 'emacs', 'neovim', 'mousepad', 'pluma',
        # Office applications
        'writer', 'calc', 'impress', 'gnumeric', 'excel', 'word', 'powerpoint',
        # IDEs and code editors
        'vscode', 'code', 'sublime', 'atom', 'pycharm', 'intellij', 'eclipse',
        'netbeans', 'brackets', 'notepad++',
        # Browsers
        'chrome', 'firefox', 'brave', 'edge', 'safari', 'chromium', 'opera',
        # Media players/editors
        'vlc', 'mpv', 'gimp', 'inkscape', 'blender', 'audacity', 'kdenlive',
        'rhythmbox', 'totem', 'spotify',
        # Development tools
        'python3', 'python', 'node', 'npm', 'java', 'javac', 'gcc', 'g++',
        'cargo', 'rustc', 'go', 'ruby', 'perl', 'php', 'pandoc',
        # Terminals and shells
        'bash', 'zsh', 'fish', 'sh', 'dash', 'gnome-terminal', 'konsole',
        'xterm', 'terminator', 'alacritty',
        # File managers
        'nautilus', 'dolphin', 'thunar', 'nemo', 'pcmanfm', 'ranger',
        # System processes
        'systemd', 'cron', 'anacron', 'rsync', 'tar', 'gzip', 'zip', 'unzip',
        'logrotate', 'updatedb', 'fc-cache', 'bleachbit',
        # Web servers and system services
        'apache2', 'httpd', 'nginx', 'mysql', 'postgres', 'ssh', 'sshd', 'mysqldump',
        # Email and communication
        'thunderbird', 'evolution', 'clipit',
        # Package management
        'apt', 'dpkg', 'fontconfig', 'steam',
        # Printing and display
        'cupsd', 'obs', 'thumbnailer',
        # Networking
        'NetworkManager', 'dockerd',
        # Archive tools
        '7z', 'unzip',
        # Version control
        'git'
    }

    # Legitimate file extensions (very low threat score)
    LEGITIMATE_EXTENSIONS = {
        # Documents
        '.pdf', '.doc', '.docx', '.odt', '.rtf', '.txt',
        # Spreadsheets  
        '.xlsx', '.xls', '.csv', '.ods',
        # Presentations
        '.ppt', '.pptx', '.odp',
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff', '.webp',
        # Media
        '.mp4', '.avi', '.mkv', '.mov', '.mp3', '.wav', '.flac', '.ogg',
        # Code
        '.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.rs', '.go',
        # Config and Logs
        '.json', '.xml', '.yaml', '.yml', '.conf', '.ini', '.toml', '.log', '.gz'
    }
    
    # Suspicious process names
    SUSPICIOUS_PROCESSES = {
        'unknown', 'malware', 'trojan', 'virus', 'backdoor', 'nc', 'netcat',
        'ncat', 'socat', 'mimikatz', 'psexec', 'powershell', 'cmd.exe',
        'suspicious', 'hack', 'exploit', '.exe', 'payload', 'reverse',
        'unknown_proc', 'cryptor', 'lockbit'  # Additional malware patterns
    }
    
    def __init__(self, window_size: int = 100, time_window_seconds: int = 300):
        """
        Initialize the feature extractor.
        
        Args:
            window_size: Number of recent operations to keep in memory
            time_window_seconds: Time window for calculating rates (default: 5 minutes)
        """
        self.window_size = window_size
        self.time_window = timedelta(seconds=time_window_seconds)
        
        # Operation history per user
        self.user_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # File access patterns
        self.file_access_count: Dict[str, int] = defaultdict(int)
        self.file_last_access: Dict[str, datetime] = {}
        self.file_sizes: Dict[str, List[int]] = defaultdict(list)
        
        # User baseline statistics
        self.user_baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # System-wide statistics
        self.global_stats = {
            'total_operations': 0,
            'operations_per_type': defaultdict(int),
            'avg_file_size': 0,
            'avg_operations_per_minute': 0
        }
        
    def extract_from_event(self, event: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a single file system event.
        
        Args:
            event: Dictionary containing:
                - timestamp: datetime or ISO string
                - user_id: str
                - operation: str (read, write, delete, rename, etc.)
                - file_path: str
                - file_size: int (bytes)
                - metadata: dict (optional additional info)
        
        Returns:
            Feature vector as numpy array
        """
        # Normalize diverse event schemas into canonical fields
        evt = self._normalize_event(event)
        # Parse timestamp
        timestamp = self._parse_timestamp(evt.get('timestamp'))
        user_id = evt.get('user_id', 'unknown')
        operation = evt.get('operation', 'read')
        file_path = evt.get('file_path', '')
        file_size = evt.get('file_size', 0)
        
        # Update history
        self._update_history(user_id, event)
        
        # Extract all feature categories
        temporal_features = self._extract_temporal_features(timestamp, user_id)
        file_features = self._extract_file_features(file_path, file_size, operation)
        behavior_features = self._extract_behavior_features(user_id, timestamp)
        security_features = self._extract_security_features(
            file_path, file_size, operation, user_id, timestamp
        )
        
        # Combine all features
        feature_vector = np.concatenate([
            temporal_features,
            file_features,
            behavior_features,
            security_features
        ])
        
        return feature_vector.astype(np.float32)
    
    def extract_from_sequence(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from a sequence of events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Feature matrix of shape (seq_len, num_features)
        """
        features = []
        for event in events:
            feature_vector = self.extract_from_event(event)
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_temporal_features(self, timestamp: datetime, user_id: str) -> np.ndarray:
        """Extract time-based features."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        is_night = 1.0 if hour < 6 or hour > 22 else 0.0
        is_business_hours = 1.0 if 9 <= hour <= 17 and day_of_week < 5 else 0.0
        
        # Calculate time since last operation for this user
        user_ops = self.user_history[user_id]
        if len(user_ops) > 0:
            last_op_time = self._parse_timestamp(user_ops[-1].get('timestamp'))
            time_since_last = (timestamp - last_op_time).total_seconds()
        else:
            time_since_last = 0.0
        
        return np.array([
            hour / 24.0,  # Normalized hour
            day_of_week / 7.0,  # Normalized day
            is_weekend,
            is_night,
            is_business_hours,
            min(time_since_last / 60.0, 100.0)  # Minutes since last op (capped at 100)
        ], dtype=np.float32)
    
    def _extract_file_features(self, file_path: str, file_size: int, operation: str) -> np.ndarray:
        """Extract file-related features."""
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        
        # File size in MB (log scale for better distribution)
        size_mb = file_size / (1024 * 1024)
        log_size = math.log1p(size_mb)  # log(1 + x) to handle zero
        
        # File type categorization
        is_executable = 1.0 if extension in self.EXECUTABLE_EXTENSIONS else 0.0
        is_document = 1.0 if extension in self.DOCUMENT_EXTENSIONS else 0.0
        is_compressed = 1.0 if extension in self.COMPRESSED_EXTENSIONS else 0.0
        is_encrypted = 1.0 if extension in self.ENCRYPTED_EXTENSIONS else 0.0
        
        # Path depth (indicator of file location)
        path_depth = len(path_obj.parts)
        
        # File name entropy (suspicious random names have high entropy)
        filename = path_obj.stem
        name_entropy = self._calculate_entropy(filename) if filename else 0.0
        
        # Operation type encoding
        operation_encoding = {
            'read': 0.0,
            'write': 0.25,
            'modify': 0.5,
            'delete': 0.75,
            'rename': 1.0
        }.get(operation.lower(), 0.5)
        
        # File access frequency
        access_freq = self.file_access_count.get(file_path, 0)
        log_access_freq = math.log1p(access_freq)
        
        return np.array([
            log_size,
            is_executable,
            is_document,
            is_compressed,
            is_encrypted,
            path_depth / 10.0,  # Normalized
            name_entropy / 8.0,  # Normalized (max entropy ~8 for random strings)
            operation_encoding,
            log_access_freq
        ], dtype=np.float32)
    
    def _extract_behavior_features(self, user_id: str, timestamp: datetime) -> np.ndarray:
        """Extract user behavior patterns."""
        user_ops = list(self.user_history[user_id])
        
        if len(user_ops) == 0:
            return np.zeros(6, dtype=np.float32)
        
        # Operations in time window
        recent_ops = [
            op for op in user_ops 
            if (timestamp - self._parse_timestamp(op.get('timestamp'))) <= self.time_window
        ]
        
        # Operation velocity (ops per minute)
        if len(recent_ops) > 1:
            time_span = (
                self._parse_timestamp(recent_ops[-1].get('timestamp')) - 
                self._parse_timestamp(recent_ops[0].get('timestamp'))
            ).total_seconds() / 60.0
            ops_per_minute = len(recent_ops) / max(time_span, 0.1)
        else:
            ops_per_minute = 0.0
        
        # Operation diversity (unique operations)
        unique_ops = len(set(op.get('operation', 'read') for op in recent_ops))
        op_diversity = unique_ops / max(len(recent_ops), 1)
        
        # File diversity (unique files accessed)
        unique_files = len(set(op.get('file_path', '') for op in recent_ops))
        file_diversity = unique_files / max(len(recent_ops), 1)
        
        # Average file size in recent operations
        recent_sizes = [op.get('file_size', 0) for op in recent_ops]
        avg_size = np.mean(recent_sizes) / (1024 * 1024) if recent_sizes else 0.0
        log_avg_size = math.log1p(avg_size)
        
        # Burstiness: std deviation of inter-operation times
        if len(recent_ops) > 2:
            timestamps = [self._parse_timestamp(op.get('timestamp')) for op in recent_ops]
            inter_times = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                          for i in range(len(timestamps)-1)]
            burstiness = np.std(inter_times) / (np.mean(inter_times) + 1e-6)
        else:
            burstiness = 0.0
        
        # Pattern regularity (compared to user baseline)
        baseline_ops_per_min = self.user_baselines[user_id].get('ops_per_minute', ops_per_minute)
        deviation = abs(ops_per_minute - baseline_ops_per_min) / (baseline_ops_per_min + 1e-6)
        
        return np.array([
            min(ops_per_minute / 10.0, 10.0),  # Capped and normalized
            op_diversity,
            file_diversity,
            log_avg_size,
            min(burstiness, 10.0),  # Capped
            min(deviation, 10.0)  # Capped
        ], dtype=np.float32)
    
    def _extract_security_features(
        self, 
        file_path: str, 
        file_size: int, 
        operation: str,
        user_id: str,
        timestamp: datetime
    ) -> np.ndarray:
        """Extract security-specific indicators."""
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        filename = path_obj.name.lower()
        file_path_lower = file_path.lower()
        
        # Ransomware indicators
        has_ransomware_extension = 1.0 if extension in self.RANSOMWARE_PATTERNS['extensions'] else 0.0
        has_ransomware_filename = 1.0 if any(pattern in filename for pattern in self.RANSOMWARE_PATTERNS['filenames']) else 0.0
        
        # Critical system path access
        is_critical_path = 1.0 if any(path in file_path_lower for path in self.CRITICAL_SYSTEM_PATHS) else 0.0
        
        # Process trust evaluation from user_history
        user_ops = list(self.user_history[user_id])
        recent_process = None
        if len(user_ops) > 0 and 'process' in user_ops[-1]:
            recent_process = str(user_ops[-1].get('process', '')).lower()
        
        # Check if process is trusted (allowlist)
        is_trusted_process = 1.0 if recent_process and any(
            trusted in recent_process for trusted in self.TRUSTED_PROCESSES
        ) else 0.0
        
        # Check if process is suspicious (blocklist)
        is_suspicious_process = 1.0 if recent_process and any(
            sus in recent_process for sus in self.SUSPICIOUS_PROCESSES
        ) else 0.0
        
        # Override: Trusted processes reduce suspicion significantly
        if is_trusted_process:
            is_suspicious_process = 0.0
        
        # Rapid file modifications (indicator of encryption)
        user_ops = list(self.user_history[user_id])
        recent_writes = [
            op for op in user_ops[-50:]  # Last 50 operations
            if op.get('operation', '').lower() in ['write', 'modify']
            and (timestamp - self._parse_timestamp(op.get('timestamp'))).total_seconds() < 60
        ]
        rapid_modifications = len(recent_writes) / 50.0
        
        # File size changes (large changes might indicate encryption)
        if file_path in self.file_sizes:
            size_history = self.file_sizes[file_path]
            if len(size_history) > 0:
                avg_prev_size = np.mean(size_history)
                size_change_ratio = abs(file_size - avg_prev_size) / (avg_prev_size + 1)
            else:
                size_change_ratio = 0.0
        else:
            size_change_ratio = 0.0
        
        # Suspicious deletion rate
        recent_deletes = [
            op for op in user_ops[-50:]
            if op.get('operation', '').lower() == 'delete'
            and (timestamp - self._parse_timestamp(op.get('timestamp'))).total_seconds() < 300
        ]
        delete_rate = len(recent_deletes) / 50.0
        
        # Mass file operation indicator (exfiltration or ransomware)
        mass_operation_score = rapid_modifications + delete_rate
        
        # Extension change frequency (renaming attacks)
        recent_renames = [
            op for op in user_ops[-50:]
            if op.get('operation', '').lower() == 'rename'
            and (timestamp - self._parse_timestamp(op.get('timestamp'))).total_seconds() < 300
        ]
        rename_rate = len(recent_renames) / 50.0
        
        # Hidden file operation (files starting with .)
        is_hidden = 1.0 if path_obj.name.startswith('.') else 0.0
        
        # Unusual access pattern (operations outside normal hours)
        hour = timestamp.hour
        is_unusual_time = 1.0 if hour < 6 or hour > 22 else 0.0
        
        # Check if file extension is legitimate
        is_legitimate_extension = 1.0 if extension in self.LEGITIMATE_EXTENSIONS else 0.0
        
        # Combine threat indicators for elevated score
        threat_multiplier = 1.0
        
        # LEGITIMATE FILES: Significantly reduce threat score
        if is_legitimate_extension > 0.5:
            threat_multiplier *= 0.2  # Reduce by 80% for legitimate file types
        
        # TRUSTED PROCESSES: Significantly reduce threat score
        if is_trusted_process > 0.5:
            threat_multiplier *= 0.05  # Reduce by 95% for trusted processes
            
        # DOUBLE PROTECTION: Both trusted process AND legitimate file
        if is_trusted_process > 0.5 and is_legitimate_extension > 0.5:
            threat_multiplier *= 0.01  # Reduce by 99% for trusted + legitimate
        
        if is_critical_path > 0.5:
            threat_multiplier *= 2.0  # Double the threat score for critical paths
        if is_suspicious_process > 0.5:
            threat_multiplier *= 2.0  # Double for suspicious processes
        if has_ransomware_extension > 0.5 or has_ransomware_filename > 0.5:
            threat_multiplier *= 3.0  # Triple for ransomware indicators
        
        return np.array([
            has_ransomware_extension,
            has_ransomware_filename,
            rapid_modifications * threat_multiplier,
            min(size_change_ratio, 10.0),
            delete_rate * threat_multiplier,
            mass_operation_score * threat_multiplier,
            rename_rate,
            is_hidden,
            is_unusual_time,
            is_critical_path,
            is_suspicious_process,
            min(threat_multiplier / 3.0, 3.0),  # Normalized threat multiplier
            is_trusted_process,  # Trusted process indicator
            is_legitimate_extension  # NEW: Legitimate file extension indicator
        ], dtype=np.float32)
    
    def _update_history(self, user_id: str, event: Dict[str, Any]):
        """Update internal statistics with new event."""
        # Store normalized event representation for downstream computations
        evt = self._normalize_event(event)
        self.user_history[user_id].append(evt)

        file_path = evt.get('file_path', '')
        if file_path:
            self.file_access_count[file_path] += 1
            self.file_last_access[file_path] = self._parse_timestamp(evt.get('timestamp'))

            file_size = evt.get('file_size', 0)
            self.file_sizes[file_path].append(file_size)
            if len(self.file_sizes[file_path]) > 10:
                self.file_sizes[file_path] = self.file_sizes[file_path][-10:]

        # Update global statistics
        self.global_stats['total_operations'] += 1
        operation = evt.get('operation', 'read')
        self.global_stats['operations_per_type'][operation] += 1
    
    def update_user_baseline(self, user_id: str):
        """Calculate and update baseline statistics for a user."""
        user_ops = list(self.user_history[user_id])
        
        if len(user_ops) < 10:
            return  # Not enough data for baseline
        
        # Calculate average operations per minute
        if len(user_ops) > 1:
            time_span = (
                self._parse_timestamp(user_ops[-1].get('timestamp')) -
                self._parse_timestamp(user_ops[0].get('timestamp'))
            ).total_seconds() / 60.0
            ops_per_minute = len(user_ops) / max(time_span, 0.1)
        else:
            ops_per_minute = 0.0
        
        # Calculate average file size
        file_sizes = [op.get('file_size', 0) for op in user_ops]
        avg_file_size = np.mean(file_sizes) / (1024 * 1024)  # MB
        
        # Store baseline
        self.user_baselines[user_id] = {
            'ops_per_minute': ops_per_minute,
            'avg_file_size': avg_file_size,
            'last_updated': datetime.now()
        }
        
        logger.debug(f"Updated baseline for user {user_id}: {self.user_baselines[user_id]}")
    
    def get_feature_names(self) -> List[str]:
        """Return names of all extracted features."""
        return [
            # Temporal (6)
            'hour_normalized', 'day_of_week_normalized', 'is_weekend',
            'is_night', 'is_business_hours', 'time_since_last_op',
            # File (9)
            'log_file_size', 'is_executable', 'is_document', 'is_compressed',
            'is_encrypted', 'path_depth', 'filename_entropy', 'operation_type',
            'log_access_frequency',
            # Behavior (6)
            'ops_per_minute', 'operation_diversity', 'file_diversity',
            'log_avg_size', 'burstiness', 'baseline_deviation',
            # Security (14) - expanded with trust and legitimacy indicators
            'has_ransomware_ext', 'has_ransomware_name', 'rapid_modifications',
            'size_change_ratio', 'delete_rate', 'mass_operation_score',
            'rename_rate', 'is_hidden', 'is_unusual_time', 'is_critical_path',
            'is_suspicious_process', 'threat_multiplier', 'is_trusted_process',
            'is_legitimate_extension'
        ]
    
    def get_num_features(self) -> int:
        """Return total number of features extracted."""
        return len(self.get_feature_names())
    
    @staticmethod
    def _parse_timestamp(timestamp: Any) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(timestamp, datetime):
            return timestamp
        # Support numeric epoch seconds (float or int)
        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(float(timestamp))
            except Exception:
                return datetime.now()
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except Exception:
                # Fallback: try parse as epoch string
                try:
                    return datetime.fromtimestamp(float(timestamp))
                except Exception:
                    return datetime.now()
        return datetime.now()

    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Map various event schemas to canonical keys used by extractor.
        Canonical keys: timestamp, user_id, operation, file_path, file_size
        """
        if not isinstance(event, dict):
            return {
                'timestamp': datetime.now(),
                'user_id': 'unknown',
                'operation': 'read',
                'file_path': '',
                'file_size': 0
            }
        # Prefer explicit fields; fallback to common alternates from synthetic generator
        ts = event.get('timestamp')
        user_id = event.get('user_id') or event.get('user') or 'unknown'
        # Map event_type to operation and normalize common verbs
        op = event.get('operation') or event.get('event_type') or 'read'
        op_norm = str(op).lower()
        # Map CREATE/MODIFY/DELETE/CHMOD/CHOWN to our set
        mapping = {
            'create': 'write',
            'modify': 'modify',
            'delete': 'delete',
            'rename': 'rename',
            'chmod': 'modify',
            'chown': 'modify'
        }
        op_norm = mapping.get(op_norm, op_norm)
        file_path = event.get('file_path') or event.get('path') or ''
        # Prefer file_size, fallback to common API key 'size'
        file_size = event.get('file_size') if event.get('file_size') is not None else event.get('size', 0)
        # Extract process name if available
        process = event.get('process') or event.get('process_name') or ''
        return {
            'timestamp': ts,
            'user_id': user_id,
            'operation': op_norm,
            'file_path': file_path,
            'file_size': file_size,
            'process': process
        }
    
    @staticmethod
    def _calculate_entropy(s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s:
            return 0.0
        
        # Count character frequencies
        char_freq = defaultdict(int)
        for char in s:
            char_freq[char] += 1
        
        # Calculate entropy
        length = len(s)
        entropy = 0.0
        for count in char_freq.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _extract_advanced_threat_features(self, file_path: str, file_size: int, 
                                        operation: str, user_id: str, timestamp: datetime) -> np.ndarray:
        """Extract advanced threat intelligence features."""
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        filename = path_obj.name.lower()
        file_path_lower = file_path.lower()
        
        user_ops = list(self.user_history[user_id])
        recent_ops = [
            op for op in user_ops[-100:]  # Last 100 operations
            if (timestamp - self._parse_timestamp(op.get('timestamp'))).total_seconds() < 300  # 5 minutes
        ]
        
        # Pattern-based threat detection
        
        # 1. File entropy analysis (encrypted/compressed files have high entropy)
        entropy_score = self._calculate_entropy(filename) / 8.0  # Normalize to 0-1
        
        # 2. Double extension detection (.pdf.exe, .doc.scr)
        parts = filename.split('.')
        has_double_extension = 1.0 if len(parts) > 2 and parts[-2] in {
            'pdf', 'doc', 'docx', 'jpg', 'png', 'txt', 'zip'
        } and parts[-1] in {'exe', 'scr', 'bat', 'cmd', 'pif'} else 0.0
        
        # 3. Suspicious file location patterns
        is_temp_execution = 1.0 if any(temp in file_path_lower for temp in [
            '/tmp/', '\\temp\\', '/var/tmp/', 'appdata/local/temp', 'downloads'
        ]) and extension in {'.exe', '.bat', '.sh', '.cmd'} else 0.0
        
        # 4. Process diversity (multiple different processes in short time)
        recent_processes = set()
        for op in recent_ops:
            if 'process' in op:
                recent_processes.add(str(op['process']).lower())
        process_diversity = min(len(recent_processes) / 10.0, 1.0)  # Normalize
        
        # 5. File access velocity (files accessed per minute)
        unique_files = set()
        for op in recent_ops:
            if 'file_path' in op:
                unique_files.add(op['file_path'])
        file_velocity = min(len(unique_files) / 20.0, 1.0)  # Normalize
        
        # 6. Cross-directory operations (accessing multiple directories)
        directories = set()
        for op in recent_ops:
            if 'file_path' in op:
                directories.add(str(Path(op['file_path']).parent))
        directory_spread = min(len(directories) / 10.0, 1.0)
        
        # 7. Privilege escalation patterns
        has_privilege_paths = 1.0 if any(priv in file_path_lower for priv in [
            '/etc/', '/root/', '/usr/bin/', '/usr/sbin/', 'system32', 'syswow64',
            'c:\\windows\\', 'c:\\program files'
        ]) else 0.0
        
        # 8. Network-related file operations (potential exfiltration)
        network_related = 1.0 if any(net in filename for net in [
            'network', 'tcp', 'udp', 'ssh', 'ftp', 'http', 'curl', 'wget', 'nc'
        ]) else 0.0
        
        # 9. Crypto/mining indicators
        crypto_indicators = 1.0 if any(crypto in filename for crypto in [
            'bitcoin', 'btc', 'ethereum', 'eth', 'mining', 'miner', 'crypto',
            'wallet', 'xmrig', 'monero', 'xmr'
        ]) else 0.0
        
        # 10. Log manipulation patterns
        log_tampering = 1.0 if (operation.lower() in ['delete', 'modify'] and 
                               any(log in file_path_lower for log in [
            '/var/log/', 'event.log', 'security.log', 'auth.log', 'syslog'
        ])) else 0.0
        
        return np.array([
            entropy_score,
            has_double_extension,  
            is_temp_execution,
            process_diversity,
            file_velocity,
            directory_spread,
            has_privilege_paths,
            network_related,
            crypto_indicators,
            log_tampering
        ], dtype=np.float32)
    
    def reset(self):
        """Reset all internal state."""
        self.user_history.clear()
        self.file_access_count.clear()
        self.file_last_access.clear()
        self.file_sizes.clear()
        self.user_baselines.clear()
        self.global_stats = {
            'total_operations': 0,
            'operations_per_type': defaultdict(int),
            'avg_file_size': 0,
            'avg_operations_per_minute': 0
        }
        logger.info("Feature extractor state reset")

#!/usr/bin/env python3
"""
SentinelZer0 Production Training Script
========================================

Trains the hybrid threat detection model using synthetic file system events.
Generates realistic normal and anomalous events for comprehensive training.

Usage:
    python train_model.py [--epochs 50] [--batch-size 32] [--gpu]
"""

import sys
import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sentinelzer0.models.hybrid_detector import HybridThreatDetector
from sentinelzer0.data.real_feature_extractor import RealFeatureExtractor
from sentinelzer0.training.real_trainer import RealWorldTrainer
from sentinelzer0.utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_events(
    num_normal: int = 4000,
    num_anomaly: int = 1000,
    seed: int = 42
) -> Tuple[List[Dict], np.ndarray]:
    """
    Generate synthetic file system events for training.
    
    Args:
        num_normal: Number of normal events
        num_anomaly: Number of anomalous events
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (events, labels)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info("="*80)
    logger.info("GENERATING SYNTHETIC TRAINING DATA")
    logger.info("="*80)
    
    events = []
    labels = []
    
    # User profiles
    users = ['alice', 'bob', 'charlie', 'admin', 'service_account']
    
    # Normal file extensions
    normal_extensions = [
        '.txt', '.pdf', '.doc', '.docx', '.xlsx', '.csv',
        '.jpg', '.png', '.gif', '.mp4', '.mp3',
        '.py', '.js', '.json', '.xml', '.html', '.css',
        '.log', '.conf', '.ini'
    ]
    
    # Ransomware extensions
    ransomware_extensions = [
        '.encrypted', '.locked', '.crypto', '.crypt',
        '.aes', '.rsa', '.cerber', '.locky', '.wannacry'
    ]
    
    # Normal process names
    normal_processes = [
        'chrome', 'firefox', 'vscode', 'python3', 'node',
        'libreoffice', 'gimp', 'vlc', 'spotify', 'terminal'
    ]
    
    # Malicious process names
    malicious_processes = [
        'ransomware.exe', 'cryptor.bin', 'malware.exe',
        'unknown_service', 'suspicious.bin', 'backdoor.exe',
        'keylogger.exe', 'trojan.exe'
    ]
    
    base_time = datetime.now() - timedelta(days=30)
    
    # ===== GENERATE NORMAL EVENTS =====
    logger.info(f"Generating {num_normal} normal events...")
    
    for i in range(num_normal):
        user = random.choice(users)
        ext = random.choice(normal_extensions)
        process = random.choice(normal_processes)
        hour = random.randint(8, 18)  # Business hours
        
        # Normal operations
        operation = random.choice(['CREATE', 'MODIFY', 'DELETE'])
        
        event = {
            'timestamp': (base_time + timedelta(hours=hour, minutes=i % 60)).timestamp(),
            'event_type': operation,
            'path': f'/home/{user}/documents/file_{i}{ext}',
            'process_name': process,
            'user': user,
            'file_size': random.randint(1024, 10*1024*1024),  # 1KB - 10MB
            'pid': random.randint(1000, 9999),
            'ppid': random.randint(500, 999),
            'extension': ext,
            'is_system': False,
            'is_hidden': False,
            'permissions': 644
        }
        
        events.append(event)
        labels.append(0)  # Normal
    
    # ===== GENERATE ANOMALOUS EVENTS =====
    logger.info(f"Generating {num_anomaly} anomalous events...")
    
    threat_start = base_time + timedelta(days=15)
    
    # Calculate threat type distribution
    num_ransomware = int(num_anomaly * 0.4)  # 40%
    num_exfiltration = int(num_anomaly * 0.2)  # 20%
    num_mass_delete = int(num_anomaly * 0.2)  # 20%
    num_privilege_esc = num_anomaly - (num_ransomware + num_exfiltration + num_mass_delete)  # 20%
    
    anomaly_idx = 0
    
    # 1. RANSOMWARE ATTACKS (40%)
    logger.info(f"  - {num_ransomware} ransomware events")
    for i in range(num_ransomware):
        user = random.choice(users[:3])  # Target regular users
        ransom_ext = random.choice(ransomware_extensions)
        malware = random.choice(malicious_processes)
        
        # Rapid file encryption pattern
        event = {
            'timestamp': (threat_start + timedelta(seconds=anomaly_idx * 2)).timestamp(),
            'event_type': random.choice(['CREATE', 'MODIFY']),
            'path': f'/home/{user}/documents/important_{i}{ransom_ext}',
            'process_name': malware,
            'user': user,
            'file_size': random.randint(100*1024, 5*1024*1024),
            'pid': random.randint(10000, 19999),
            'ppid': 1,  # Spawned from init (suspicious)
            'extension': ransom_ext,
            'is_system': False,
            'is_hidden': False,
            'permissions': 666  # World writable (suspicious)
        }
        
        events.append(event)
        labels.append(1)  # Anomaly
        anomaly_idx += 1
    
    # 2. DATA EXFILTRATION (20%)
    logger.info(f"  - {num_exfiltration} data exfiltration events")
    for i in range(num_exfiltration):
        user = random.choice(users)
        
        # Large file reads + network transfer
        event = {
            'timestamp': (threat_start + timedelta(hours=5, seconds=anomaly_idx * 3)).timestamp(),
            'event_type': 'MODIFY',
            'path': f'/home/{user}/sensitive/database_{i}.sql',
            'process_name': random.choice(['curl', 'wget', 'netcat', 'unknown_tool']),
            'user': user,
            'file_size': random.randint(100*1024*1024, 2*1024*1024*1024),  # 100MB - 2GB
            'pid': random.randint(20000, 29999),
            'ppid': 1,
            'extension': '.sql',
            'is_system': False,
            'is_hidden': False,
            'permissions': 600
        }
        
        events.append(event)
        labels.append(1)
        anomaly_idx += 1
    
    # 3. MASS DELETION (20%)
    logger.info(f"  - {num_mass_delete} mass deletion events")
    for i in range(num_mass_delete):
        user = random.choice(users[:3])
        
        # Rapid file deletion
        event = {
            'timestamp': (threat_start + timedelta(hours=10, seconds=anomaly_idx)).timestamp(),
            'event_type': 'DELETE',
            'path': f'/home/{user}/important/critical_file_{i}.txt',
            'process_name': random.choice(malicious_processes),
            'user': user,
            'file_size': 0,
            'pid': random.randint(30000, 39999),
            'ppid': 1,
            'extension': random.choice(['.txt', '.pdf', '.doc']),
            'is_system': False,
            'is_hidden': False,
            'permissions': 644
        }
        
        events.append(event)
        labels.append(1)
        anomaly_idx += 1
    
    # 4. PRIVILEGE ESCALATION (20%)
    logger.info(f"  - {num_privilege_esc} privilege escalation events")
    for i in range(num_privilege_esc):
        # Unauthorized system file access
        event = {
            'timestamp': (threat_start + timedelta(hours=15, seconds=anomaly_idx * 5)).timestamp(),
            'event_type': random.choice(['MODIFY', 'CHMOD', 'CHOWN']),
            'path': random.choice(['/etc/passwd', '/etc/shadow', '/etc/sudoers', '/root/.ssh/authorized_keys']),
            'process_name': random.choice(malicious_processes),
            'user': random.choice(['www-data', 'nobody', 'service_account']),  # Low-privilege users
            'file_size': random.randint(1024, 10*1024),
            'pid': random.randint(40000, 49999),
            'ppid': 1,
            'extension': '',
            'is_system': True,  # CRITICAL: System file access
            'is_hidden': False,
            'permissions': 600
        }
        
        events.append(event)
        labels.append(1)
        anomaly_idx += 1
    
    # Shuffle to mix normal and anomalous
    combined = list(zip(events, labels))
    random.shuffle(combined)
    events, labels = zip(*combined)
    
    events = list(events)
    labels_array = np.array(labels)
    
    # Verify distribution
    logger.info("="*80)
    logger.info(f"DATASET SUMMARY:")
    logger.info(f"  Total events: {len(events)}")
    logger.info(f"  Normal: {(labels_array == 0).sum()} ({(labels_array == 0).sum()/len(labels_array)*100:.1f}%)")
    logger.info(f"  Anomalous: {(labels_array == 1).sum()} ({(labels_array == 1).sum()/len(labels_array)*100:.1f}%)")
    logger.info("="*80)
    
    return events, labels_array


def train_model(
    num_epochs: int = 50,
    batch_size: int = 32,
    sequence_length: int = 64,
    learning_rate: float = 0.001,
    use_gpu: bool = False,
    save_path: str = 'models/production/trained_model.pt'
) -> Dict:
    """
    Train the hybrid threat detection model.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        sequence_length: Length of event sequences
        learning_rate: Learning rate
        use_gpu: Whether to use GPU
        save_path: Path to save trained model
        
    Returns:
        Training metrics and history
    """
    logger.info("="*80)
    logger.info("SENTINELZER0 MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Device: {'GPU' if use_gpu else 'CPU'}")
    logger.info("="*80)
    
    # Set device
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    if use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but not available, using CPU")
    
    # Generate synthetic data
    all_events, all_labels = generate_synthetic_events(
        num_normal=4000,
        num_anomaly=1000
    )
    
    # Split data: 70% train, 15% val, 15% test
    n_total = len(all_events)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_events = all_events[:n_train]
    train_labels = all_labels[:n_train]
    
    val_events = all_events[n_train:n_train+n_val]
    val_labels = all_labels[n_train:n_train+n_val]
    
    test_events = all_events[n_train+n_val:]
    test_labels = all_labels[n_train+n_val:]
    
    logger.info(f"Data split:")
    logger.info(f"  Training: {len(train_events)} events")
    logger.info(f"  Validation: {len(val_events)} events")
    logger.info(f"  Test: {len(test_events)} events")
    
    # Initialize components
    logger.info("Initializing model and trainer...")
    
    feature_extractor = RealFeatureExtractor()
    
    model = HybridThreatDetector(
        input_size=30,  # Feature dimension
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    
    # Move model to device
    model = model.to(device)
    
    trainer = RealWorldTrainer(
        model=model,
        feature_extractor=feature_extractor,
        learning_rate=learning_rate,
        weight_decay=0.0001,
        class_weight_positive=3.0,  # Weight anomaly class more (imbalanced data)
        patience=10,
        checkpoint_dir='./checkpoints'
    )
    
    # Train model
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    history = trainer.train_from_real_data(
        train_events=train_events,
        val_events=val_events,
        train_labels=train_labels,
        val_labels=val_labels,
        num_epochs=num_epochs,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    # Save final model
    logger.info("="*80)
    logger.info("SAVING TRAINED MODEL")
    logger.info("="*80)
    
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_extractor_config': {
            'input_size': 30,
            'sequence_length': sequence_length
        },
        'model_config': {
            'input_size': 30,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'sequence_length': sequence_length
        },
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to: {save_path}")
    
    # Evaluate on test set
    logger.info("="*80)
    logger.info("TEST SET EVALUATION")
    logger.info("="*80)
    
    test_features = trainer._prepare_sequences(test_events, sequence_length)
    test_labels_aligned = trainer._align_labels(test_labels, len(test_features))
    
    # Create test loader
    test_loader = trainer._create_dataloader(
        test_features, test_labels_aligned, batch_size, shuffle=False
    )
    
    # Evaluate
    model.eval()
    model = model.to(device)  # Ensure model is on correct device
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_labels.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = (all_preds == all_targets).mean()
    precision = ((all_preds == 1) & (all_targets == 1)).sum() / max(1, (all_preds == 1).sum())
    recall = ((all_preds == 1) & (all_targets == 1)).sum() / max(1, (all_targets == 1).sum())
    f1 = 2 * precision * recall / max(0.001, precision + recall)
    
    logger.info(f"Test Set Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    
    # Save metrics
    metrics_path = save_path_obj.parent / 'training_metrics.json'
    metrics = {
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'best_val_f1': history.get('best_val_f1', 0.0),
        'best_epoch': history.get('best_epoch', 0),
        'total_epochs': num_epochs,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_path}")
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Model ready for production use: {save_path}")
    logger.info(f"Update your API to load this model for real threat detection.")
    logger.info("="*80)
    
    return history


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train SentinelZer0 threat detection model'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--sequence-length', type=int, default=64,
        help='Event sequence length (default: 64)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Use GPU if available'
    )
    parser.add_argument(
        '--save-path', type=str, default='models/production/trained_model.pt',
        help='Path to save trained model (default: models/production/trained_model.pt)'
    )
    
    args = parser.parse_args()
    
    # Train model
    try:
        history = train_model(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            learning_rate=args.learning_rate,
            use_gpu=args.gpu,
            save_path=args.save_path
        )
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

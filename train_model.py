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
    num_normal: int = 8000,
    num_anomaly: int = 1000,
    seed: int = 42
) -> Tuple[List[Dict], np.ndarray]:
    """
    Generate synthetic file system events for training.
    
    Args:
        num_normal: Number of normal events (increased for better balance)
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
    
    # Normal file extensions (categorized for realistic patterns)
    document_extensions = ['.pdf', '.doc', '.docx', '.odt', '.txt', '.rtf']
    spreadsheet_extensions = ['.xlsx', '.xls', '.csv', '.ods']
    presentation_extensions = ['.ppt', '.pptx', '.odp']
    image_extensions = ['.jpg', '.png', '.gif', '.bmp', '.svg']
    media_extensions = ['.mp4', '.mp3', '.avi', '.mkv', '.wav']
    code_extensions = ['.py', '.js', '.java', '.cpp', '.rs', '.go', '.html', '.css']
    config_extensions = ['.json', '.xml', '.yaml', '.yml', '.conf', '.ini']
    
    normal_extensions = (document_extensions + spreadsheet_extensions + 
                        presentation_extensions + image_extensions + 
                        media_extensions + code_extensions + config_extensions)
    
    # Ransomware extensions
    ransomware_extensions = [
        '.encrypted', '.locked', '.crypto', '.crypt',
        '.aes', '.rsa', '.cerber', '.locky', '.wannacry'
    ]
    
    # Normal process names (TRUSTED APPLICATIONS)
    document_processes = ['evince', 'okular', 'libreoffice', 'abiword', 'gedit', 'kate', 'nano', 'vim']
    spreadsheet_processes = ['libreoffice', 'gnumeric', 'excel']
    ide_processes = ['vscode', 'sublime', 'atom', 'pycharm', 'intellij', 'eclipse']
    browser_processes = ['chrome', 'firefox', 'brave', 'edge', 'safari']
    media_processes = ['vlc', 'mpv', 'gimp', 'inkscape', 'blender', 'audacity']
    terminal_processes = ['bash', 'zsh', 'python3', 'node', 'java', 'gcc']
    system_processes = ['nautilus', 'dolphin', 'thunar', 'systemd', 'cron']
    
    normal_processes = (document_processes + spreadsheet_processes + ide_processes + 
                       browser_processes + media_processes + terminal_processes + system_processes)
    
    # Malicious process names
    malicious_processes = [
        'ransomware.exe', 'cryptor.bin', 'malware.exe',
        'unknown_service', 'suspicious.bin', 'backdoor.exe',
        'keylogger.exe', 'trojan.exe'
    ]
    
    base_time = datetime.now() - timedelta(days=30)
    
    # ===== GENERATE ADVANCED NORMAL EVENTS =====
    logger.info(f"Generating {num_normal} normal events with realistic patterns...")
    
    # User behavior patterns (more sophisticated)
    user_profiles = {
        'alice': {'work_start': 9, 'work_end': 17, 'productivity': 0.8, 'tech_savvy': 0.6},
        'bob': {'work_start': 8, 'work_end': 18, 'productivity': 0.9, 'tech_savvy': 0.9},
        'charlie': {'work_start': 10, 'work_end': 16, 'productivity': 0.7, 'tech_savvy': 0.4},
        'admin': {'work_start': 7, 'work_end': 20, 'productivity': 0.9, 'tech_savvy': 1.0},
        'service_account': {'work_start': 0, 'work_end': 23, 'productivity': 1.0, 'tech_savvy': 1.0}
    }
    
    # Realistic workflow patterns
    workflow_templates = {
        'document_editing': {
            'duration_minutes': (15, 120), 'operations': ['CREATE', 'MODIFY', 'MODIFY', 'MODIFY'],
            'processes': document_processes, 'extensions': document_extensions
        },
        'development_session': {
            'duration_minutes': (30, 240), 'operations': ['CREATE', 'MODIFY'] * 10,
            'processes': ide_processes + terminal_processes, 'extensions': code_extensions
        },
        'data_analysis': {
            'duration_minutes': (45, 180), 'operations': ['CREATE', 'MODIFY', 'MODIFY'],
            'processes': ['python3', 'jupyter', 'rstudio', 'excel'], 'extensions': ['.csv', '.xlsx', '.py', '.r']
        },
        'media_processing': {
            'duration_minutes': (20, 90), 'operations': ['CREATE', 'MODIFY'],
            'processes': media_processes, 'extensions': image_extensions + media_extensions
        }
    }
    
    # Generate normal events with realistic temporal clustering
    current_time = base_time
    event_clusters = []  # Group related events
    
    for i in range(num_normal):
        user = random.choice(users)
        profile = user_profiles[user]
        
        # Realistic time patterns
        if random.random() < 0.8:  # 80% during work hours
            hour = random.randint(profile['work_start'], profile['work_end'])
        else:  # 20% outside work hours
            hour = random.choice([random.randint(0, 6), random.randint(22, 23)])
        
        # Choose workflow pattern
        workflow = random.choice(list(workflow_templates.keys()))
        template = workflow_templates[workflow]
        
        # Generate file type and matching process (more sophisticated)
        if workflow == 'document_editing':
            ext = random.choice(template['extensions'])
            # Ensure trusted processes are matched with legitimate extensions
            process = random.choice(template['processes'])
            # Simulate realistic document editing patterns
            if i % 4 == 0:  # Start new document
                operation = 'CREATE'
                size = random.randint(1024, 10*1024)  # Small initial size
            else:  # Edit existing document
                operation = 'MODIFY'
                size = random.randint(50*1024, 2*1024*1024)  # Growing size
            path_dir = random.choice(['documents', 'work', 'reports', 'papers'])
            
        elif workflow == 'development_session':
            ext = random.choice(template['extensions'])
            process = random.choice(template['processes'])
            operation = random.choices(['MODIFY', 'CREATE', 'DELETE'], weights=[0.7, 0.25, 0.05])[0]
            size = random.randint(512, 100*1024)  # Code files are usually small
            path_dir = random.choice(['projects', 'code', 'dev', 'src', 'workspace'])
            
        elif workflow == 'data_analysis':
            ext = random.choice(template['extensions'])
            process = random.choice(template['processes'])
            operation = random.choices(['MODIFY', 'CREATE'], weights=[0.8, 0.2])[0]
            size = random.randint(10*1024, 50*1024*1024)  # Data files can be large
            path_dir = random.choice(['data', 'analysis', 'datasets', 'research'])
            
        else:  # media_processing
            ext = random.choice(template['extensions'])
            process = random.choice(template['processes'])
            operation = random.choices(['CREATE', 'MODIFY', 'DELETE'], weights=[0.5, 0.4, 0.1])[0]
            # Realistic media file sizes
            if ext in ['.jpg', '.png', '.gif']:
                size = random.randint(50*1024, 10*1024*1024)  # 50KB - 10MB
            else:  # Video files
                size = random.randint(10*1024*1024, 500*1024*1024)  # 10MB - 500MB
            path_dir = random.choice(['pictures', 'videos', 'media', 'downloads'])
        
        # Add temporal clustering (simulate work sessions)
        if i % 10 == 0:  # Start new session
            session_offset = random.randint(0, 30)  # Minutes within hour
        else:
            session_offset = (session_offset + random.randint(1, 5)) % 60
        
        # Realistic file naming patterns
        if workflow == 'document_editing':
            filename = f"{random.choice(['report', 'document', 'memo', 'notes', 'draft'])}_{random.randint(1, 100)}{ext}"
        elif workflow == 'development_session':
            filename = f"{random.choice(['main', 'utils', 'config', 'test', 'app'])}_{random.randint(1, 50)}{ext}"
        elif workflow == 'data_analysis':
            filename = f"{random.choice(['dataset', 'analysis', 'results', 'model'])}_{random.randint(1, 20)}{ext}"
        else:
            filename = f"{random.choice(['image', 'video', 'photo', 'clip'])}_{random.randint(1, 1000)}{ext}"
        
        event = {
            'timestamp': (base_time + timedelta(days=i//200, hours=hour, minutes=session_offset)).timestamp(),
            'event_type': operation,
            'path': f'/home/{user}/{path_dir}/{filename}',
            'process_name': process,
            'user': user,
            'file_size': size,
            'pid': random.randint(1000, 9999),
            'ppid': random.randint(500, 999),
            'extension': ext,
            'is_system': False,
            'is_hidden': random.random() < 0.05,  # 5% hidden files (config files)
            'permissions': random.choice([644, 755, 600]),  # Realistic permissions
            'workflow_type': workflow,
            'user_profile': profile
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
    
    # 1. ADVANCED RANSOMWARE ATTACKS (40%)
    logger.info(f"  - {num_ransomware} advanced ransomware events")
    ransomware_families = {
        'lockbit': {'extensions': ['.lockbit', '.locked'], 'processes': ['lockbit.exe', 'encrypt.exe']},
        'wannacry': {'extensions': ['.wannacry', '.WNCRY'], 'processes': ['wannacry.exe', 'tasksche.exe']},
        'ryuk': {'extensions': ['.ryuk', '.ryk'], 'processes': ['ryuk.exe', 'rykenc.exe']},
        'maze': {'extensions': ['.maze', '.encrypted'], 'processes': ['maze.exe', 'enc.exe']}
    }
    
    for i in range(num_ransomware):
        user = random.choice(users[:3])  # Target regular users
        family = random.choice(list(ransomware_families.keys()))
        family_info = ransomware_families[family]
        
        # Simulate realistic ransomware behavior patterns
        attack_phase = i % 4  # 4 phases of ransomware attack
        
        if attack_phase == 0:  # Initial infection
            event = {
                'timestamp': (threat_start + timedelta(minutes=anomaly_idx * 0.5)).timestamp(),
                'event_type': 'CREATE',
                'path': f'/tmp/{random.choice(["update", "install", "setup"])}.exe',
                'process_name': random.choice(['svchost.exe', 'winlogon.exe', 'explorer.exe']),
                'user': user,
                'file_size': random.randint(500*1024, 2*1024*1024),
                'pid': random.randint(10000, 19999),
                'ppid': 1,
                'extension': '.exe',
                'is_system': False,
                'is_hidden': True,
                'permissions': 777,
                'attack_phase': 'infection'
            }
        elif attack_phase == 1:  # Reconnaissance
            event = {
                'timestamp': (threat_start + timedelta(minutes=anomaly_idx * 0.5 + 1)).timestamp(),
                'event_type': 'MODIFY',
                'path': random.choice(['/etc/passwd', '/home/.ssh/known_hosts', f'/home/{user}/.bash_history']),
                'process_name': random.choice(family_info['processes']),
                'user': user,
                'file_size': random.randint(1024, 10*1024),
                'pid': random.randint(10000, 19999),
                'ppid': random.randint(10000, 19999),
                'extension': '',
                'is_system': True,
                'is_hidden': False,
                'permissions': 600,
                'attack_phase': 'reconnaissance'
            }
        elif attack_phase == 2:  # Lateral movement
            event = {
                'timestamp': (threat_start + timedelta(minutes=anomaly_idx * 0.5 + 2)).timestamp(),
                'event_type': 'CREATE',
                'path': f'/home/{random.choice(users)}/shared/{random.choice(["backup", "important", "data"])}{random.choice(family_info["extensions"])}',
                'process_name': random.choice(family_info['processes']),
                'user': 'admin',  # Privilege escalation
                'file_size': random.randint(10*1024*1024, 100*1024*1024),
                'pid': random.randint(10000, 19999),
                'ppid': 1,
                'extension': random.choice(family_info['extensions']),
                'is_system': False,
                'is_hidden': False,
                'permissions': 666,
                'attack_phase': 'lateral_movement'
            }
        else:  # Mass encryption
            original_file = random.choice(['document', 'photo', 'video', 'database'])
            event = {
                'timestamp': (threat_start + timedelta(minutes=anomaly_idx * 0.5 + 3)).timestamp(),
                'event_type': random.choice(['MODIFY', 'CREATE']),
                'path': f'/home/{user}/{random.choice(["documents", "pictures", "data"])}/{original_file}_{i}{random.choice(family_info["extensions"])}',
                'process_name': random.choice(family_info['processes']),
                'user': user,
                'file_size': random.randint(100*1024, 50*1024*1024),
                'pid': random.randint(10000, 19999),
                'ppid': random.randint(10000, 19999),
                'extension': random.choice(family_info['extensions']),
                'is_system': False,
                'is_hidden': False,
                'permissions': 666,
                'attack_phase': 'encryption',
                'ransomware_family': family
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
    learning_rate: float = 0.005,
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
    
    # Generate synthetic data (increased normal events for better balance)
    all_events, all_labels = generate_synthetic_events(
        num_normal=12000,  # Increase normal events for better balance
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
    num_features = feature_extractor.get_num_features()
    logger.info(f"Feature extractor produces {num_features} features")
    
    model = HybridThreatDetector(
        input_size=num_features,  # Feature dimension (33 with new features)
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
            'input_size': num_features,
            'sequence_length': sequence_length
        },
        'model_config': {
            'input_size': num_features,
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

    decision_threshold = history.get('decision_threshold', trainer.decision_threshold)
    logger.info("Using decision threshold: %.3f", decision_threshold)

    all_scores: List[float] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            scores = outputs.detach().cpu().numpy().flatten()
            all_scores.extend(scores)
            all_targets.extend(batch_labels.numpy().flatten())

    score_array = np.array(all_scores, dtype=np.float32)
    target_array = np.array(all_targets, dtype=np.int32)

    test_metrics = trainer._calculate_classification_metrics(
        target_array,
        score_array,
        threshold=decision_threshold
    )

    logger.info("Test Set Results:")
    logger.info("  Accuracy: %.4f", test_metrics['accuracy'])
    logger.info("  Precision: %.4f", test_metrics['precision'])
    logger.info("  Recall: %.4f", test_metrics['recall'])
    logger.info("  F1-Score: %.4f", test_metrics['f1'])
    logger.info("  ROC-AUC: %.4f", test_metrics['roc_auc'])
    logger.info("  PR-AUC: %.4f", test_metrics['pr_auc'])
    logger.info(
        "  Score stats -> min: %.4f, 25%%: %.4f, median: %.4f, 75%%: %.4f, max: %.4f",
        float(np.min(score_array)) if score_array.size else 0.0,
        float(np.percentile(score_array, 25)) if score_array.size else 0.0,
        float(np.median(score_array)) if score_array.size else 0.0,
        float(np.percentile(score_array, 75)) if score_array.size else 0.0,
        float(np.max(score_array)) if score_array.size else 0.0
    )

    # Save metrics
    metrics_path = save_path_obj.parent / 'training_metrics.json'
    training_history = history.get('history', {})
    best_metrics_summary = history.get('best_metrics', {})
    best_val_metrics = best_metrics_summary.get('best_val_metrics', {})

    metrics = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'test_roc_auc': float(test_metrics['roc_auc']),
        'test_pr_auc': float(test_metrics['pr_auc']),
        'best_val_f1': float(best_val_metrics.get('f1', training_history.get('best_val_f1', 0.0))),
        'best_val_precision': float(best_val_metrics.get('precision', training_history.get('best_val_precision', 0.0))),
        'best_val_recall': float(best_val_metrics.get('recall', training_history.get('best_val_recall', 0.0))),
        'best_epoch': int(best_metrics_summary.get('best_epoch', training_history.get('best_epoch', 0) or 0)),
        'total_epochs': int(len(training_history.get('train_loss', []))),
        'decision_threshold': float(decision_threshold),
        'training_time_seconds': float(history.get('training_time', 0.0)),
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
        '--learning-rate', type=float, default=0.005,
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

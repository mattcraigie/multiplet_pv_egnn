"""
Basic Training Experiment for 3D Parity Violation Detection.

This module provides a basic training pipeline with visualization support for
the Frame-Aligned GNN classifier on parity violation detection.

Features:
- Training loop with BCE loss
- Validation and test evaluation
- Accuracy and loss tracking
- Bootstrap-based statistical test for parity violation detection
- Visualization of dataset and training convergence

Usage:
    python -m experiments.basic_train --mode full
    python -m experiments.basic_train --mode main --visualize
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from parent package - these work when running as a module (-m experiments.basic_train)
# or when the parent directory is in PYTHONPATH
try:
    from data import ParityViolationDataset, ParitySymmetricDataset, MultiHopParityViolationDataset
    from models.model import ParityViolationEGNN, MultiHopParityViolationEGNN
    from models.frame_aligned_model import FrameAlignedPVClassifier, MultiHopFrameAlignedPVClassifier
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import ParityViolationDataset, ParitySymmetricDataset, MultiHopParityViolationDataset
    from models.model import ParityViolationEGNN, MultiHopParityViolationEGNN
    from models.frame_aligned_model import FrameAlignedPVClassifier, MultiHopFrameAlignedPVClassifier


# Model type constants
MODEL_TYPE_EGNN = 'egnn'
MODEL_TYPE_FRAME_ALIGNED = 'frame_aligned'
MODEL_TYPE_MULTI_HOP_EGNN = 'multi_hop_egnn'
MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED = 'multi_hop_frame_aligned'
# Default to the new Frame-Aligned model
DEFAULT_MODEL_TYPE = MODEL_TYPE_FRAME_ALIGNED


# Seed generation constants for dataset independence
SEED_MULTIPLIER = 1000
SEED_OFFSET_TRAIN = 0
SEED_OFFSET_VAL = 1
SEED_OFFSET_TEST = 2
SEED_OFFSET_BOOTSTRAP = 3


def multi_hop_collate_fn(batch):
    """
    Custom collate function for multi-hop graphs with variable edge counts.
    """
    positions_list = []
    angles_list = []
    node_features_list = []
    edge_index_list = []
    special_pair_list = []
    labels_list = []
    batch_list = []
    
    node_offset = 0
    
    for i, sample in enumerate(batch):
        n_nodes = sample['positions'].shape[0]
        
        positions_list.append(sample['positions'])
        angles_list.append(sample['angles'])
        node_features_list.append(sample['node_features'])
        edge_index_list.append(sample['edge_index'] + node_offset)
        special_pair_list.append(sample['special_pair'] + node_offset)
        labels_list.append(sample['label'])
        batch_list.append(torch.full((n_nodes,), i, dtype=torch.long))
        
        node_offset += n_nodes
    
    return {
        'positions': torch.cat(positions_list, dim=0),
        'angles': torch.cat(angles_list, dim=0),
        'node_features': torch.cat(node_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'special_pair': torch.stack(special_pair_list, dim=0),
        'label': torch.stack(labels_list, dim=0),
        'batch': torch.cat(batch_list, dim=0)
    }


def is_multi_hop_model(model_type):
    """Check if model type is a multi-hop variant."""
    return model_type in [MODEL_TYPE_MULTI_HOP_EGNN, MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED]


def train_epoch(model, dataloader, optimizer, criterion, device, model_type=DEFAULT_MODEL_TYPE):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        
        if model_type == MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED:
            positions = batch['positions'].to(device)
            angles = batch['angles'].to(device)
            edge_index = batch['edge_index'].to(device)
            batch_idx = batch['batch'].to(device)
            logits = model(positions, angles, edge_index, batch_idx)
        elif model_type == MODEL_TYPE_MULTI_HOP_EGNN:
            positions = batch['positions'].to(device)
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            batch_idx = batch['batch'].to(device)
            logits = model(positions, node_features, edge_index, batch_idx)
        elif model_type == MODEL_TYPE_FRAME_ALIGNED:
            positions = batch['positions'].to(device)
            angles = batch['angles'].to(device)
            logits = model(positions, angles)
        else:
            node_features = batch['node_features'].to(device)
            edge_distance_3d = batch['edge_distance_3d'].to(device)
            edge_delta_z = batch['edge_delta_z'].to(device)
            edge_sin_2delta_phi = batch['edge_sin_2delta_phi'].to(device)
            logits = model(node_features, edge_distance_3d, edge_delta_z, edge_sin_2delta_phi)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device, model_type=DEFAULT_MODEL_TYPE):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)
            
            if model_type == MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED:
                positions = batch['positions'].to(device)
                angles = batch['angles'].to(device)
                edge_index = batch['edge_index'].to(device)
                batch_idx = batch['batch'].to(device)
                logits = model(positions, angles, edge_index, batch_idx)
            elif model_type == MODEL_TYPE_MULTI_HOP_EGNN:
                positions = batch['positions'].to(device)
                node_features = batch['node_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                batch_idx = batch['batch'].to(device)
                logits = model(positions, node_features, edge_index, batch_idx)
            elif model_type == MODEL_TYPE_FRAME_ALIGNED:
                positions = batch['positions'].to(device)
                angles = batch['angles'].to(device)
                logits = model(positions, angles)
            else:
                node_features = batch['node_features'].to(device)
                edge_distance_3d = batch['edge_distance_3d'].to(device)
                edge_delta_z = batch['edge_delta_z'].to(device)
                edge_sin_2delta_phi = batch['edge_sin_2delta_phi'].to(device)
                logits = model(node_features, edge_distance_3d, edge_delta_z, edge_sin_2delta_phi)
            
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


class EarlyStopping:
    """Early stopping helper to track validation loss and stop training when loss converges."""
    
    def __init__(self, patience: int = None, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
        self.enabled = patience is not None
    
    def __call__(self, val_loss: float, model) -> bool:
        if not self.enabled:
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.patience
    
    def restore_best_model(self, model):
        """Restore the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def create_model(model_type, hidden_dim, n_layers, num_slots=8, num_hops=2, readout_dim=32):
    """Create a model based on the model type."""
    if model_type == MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED:
        return MultiHopFrameAlignedPVClassifier(
            num_slots=num_slots,
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            readout_dim=readout_dim
        )
    elif model_type == MODEL_TYPE_MULTI_HOP_EGNN:
        return MultiHopParityViolationEGNN(
            node_input_dim=2,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )
    elif model_type == MODEL_TYPE_FRAME_ALIGNED:
        return FrameAlignedPVClassifier(
            num_slots=num_slots,
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            readout_dim=readout_dim
        )
    else:
        return ParityViolationEGNN(
            node_input_dim=2,
            edge_input_dim=3,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )


def run_experiment(
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1000,
    alpha: float = 0.3,
    f_pv: float = 1.0,
    hidden_dim: int = 16,
    n_layers: int = 2,
    batch_size: int = 64,
    n_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
    early_stopping_patience: int = None,
    early_stopping_min_delta: float = 1e-4,
    model_type: str = DEFAULT_MODEL_TYPE,
    num_slots: int = 8,
    num_hops: int = 2
):
    """Run a complete training experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ParityViolationDataset(n_train, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TRAIN)
    val_dataset = ParityViolationDataset(n_val, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_VAL)
    test_dataset = ParityViolationDataset(n_test, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TEST)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_slots=num_slots,
        num_hops=num_hops
    ).to(device)
    
    if verbose:
        print(f"Using model type: {model_type}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Loss history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta
    )
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_metrics = evaluate(model, val_loader, criterion, device, model_type)
        
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
        
        if early_stopping(val_metrics['loss'], model):
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {early_stopping_patience} epochs)")
            early_stopping.restore_best_model(model)
            break
        
        if verbose:
            print(f"Epoch {epoch + 1:3d}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_metrics['loss']:.4f}, "
                  f"Val Acc = {val_metrics['accuracy']:.4f}")
    
    # Final evaluation
    test_metrics = evaluate(model, test_loader, criterion, device, model_type)
    if verbose:
        print(f"\nTest Results: "
              f"Loss = {test_metrics['loss']:.4f}, "
              f"Accuracy = {test_metrics['accuracy']:.4f}")
    
    return {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'best_val_accuracy': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'epochs_trained': len(train_losses)
    }


def run_control_experiment(
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1000,
    hidden_dim: int = 16,
    n_layers: int = 2,
    batch_size: int = 64,
    n_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
    model_type: str = DEFAULT_MODEL_TYPE,
    num_slots: int = 8,
    num_hops: int = 2
):
    """Run control experiment with parity-symmetric data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print("Running CONTROL experiment (no parity violation)")
    
    # Create parity-symmetric datasets
    train_dataset = ParitySymmetricDataset(n_train, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TRAIN)
    val_dataset = ParitySymmetricDataset(n_val, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_VAL)
    test_dataset = ParitySymmetricDataset(n_test, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TEST)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_slots=num_slots,
        num_hops=num_hops
    ).to(device)
    
    if verbose:
        print(f"Using model type: {model_type}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Loss history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_metrics = evaluate(model, val_loader, criterion, device, model_type)
        
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        if verbose:
            print(f"Epoch {epoch + 1:3d}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_metrics['loss']:.4f}, "
                  f"Val Acc = {val_metrics['accuracy']:.4f}")
    
    # Final evaluation
    test_metrics = evaluate(model, test_loader, criterion, device, model_type)
    if verbose:
        print(f"\nControl Test Results: "
              f"Loss = {test_metrics['loss']:.4f}, "
              f"Accuracy = {test_metrics['accuracy']:.4f}")
        print("(Expected: ~0.5 accuracy for parity-symmetric data)")
    
    return {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


def run_statistical_test(
    n_seeds: int = 5,
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1000,
    alpha: float = 0.3,
    f_pv: float = 1.0,
    n_epochs: int = 20,
    verbose: bool = True,
    model_type: str = DEFAULT_MODEL_TYPE,
    num_slots: int = 8,
    num_hops: int = 2
):
    """Run multiple experiments with different seeds for statistical robustness."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running statistical test with {n_seeds} seeds")
        print(f"{'='*60}\n")
    
    accuracies = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"\n--- Seed {seed} ---")
        results = run_experiment(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            alpha=alpha,
            f_pv=f_pv,
            n_epochs=n_epochs,
            seed=seed,
            verbose=verbose,
            model_type=model_type,
            num_slots=num_slots,
            num_hops=num_hops
        )
        accuracies.append(results['test_accuracy'])
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Statistical Test Results (alpha={alpha}, f_pv={f_pv}):")
        print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Individual: {[f'{a:.4f}' for a in accuracies]}")
        if mean_acc > 0.55:
            print(f"  CONCLUSION: Parity violation DETECTED")
        else:
            print(f"  CONCLUSION: No significant parity violation detected")
        print(f"{'='*60}")
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'accuracies': accuracies
    }


def bootstrap_confidence_test(
    model,
    dataloader,
    device,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    null_accuracy: float = 0.5,
    bootstrap_seed: int = None,
    verbose: bool = True,
    model_type: str = DEFAULT_MODEL_TYPE
):
    """
    Perform bootstrap resampling to compute confidence intervals for accuracy.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)
            
            if model_type == MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED:
                positions = batch['positions'].to(device)
                angles = batch['angles'].to(device)
                edge_index = batch['edge_index'].to(device)
                batch_idx = batch['batch'].to(device)
                logits = model(positions, angles, edge_index, batch_idx)
            elif model_type == MODEL_TYPE_MULTI_HOP_EGNN:
                positions = batch['positions'].to(device)
                node_features = batch['node_features'].to(device)
                edge_index = batch['edge_index'].to(device)
                batch_idx = batch['batch'].to(device)
                logits = model(positions, node_features, edge_index, batch_idx)
            elif model_type == MODEL_TYPE_FRAME_ALIGNED:
                positions = batch['positions'].to(device)
                angles = batch['angles'].to(device)
                logits = model(positions, angles)
            else:
                node_features = batch['node_features'].to(device)
                edge_distance_3d = batch['edge_distance_3d'].to(device)
                edge_delta_z = batch['edge_delta_z'].to(device)
                edge_sin_2delta_phi = batch['edge_sin_2delta_phi'].to(device)
                logits = model(node_features, edge_distance_3d, edge_delta_z, edge_sin_2delta_phi)
            
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    n_samples = len(all_predictions)
    
    correct = (all_predictions == all_labels).astype(float)
    point_accuracy = np.mean(correct)
    
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_accuracies = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_correct = correct[indices]
        bootstrap_accuracies.append(np.mean(bootstrap_correct))
    
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_accuracies, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_accuracies, 100 * (1 - alpha / 2))
    
    bootstrap_centered = bootstrap_accuracies - point_accuracy + null_accuracy
    p_value = np.mean(bootstrap_centered >= point_accuracy)
    
    detection_confidence = np.mean(bootstrap_accuracies > null_accuracy)
    parity_violation_detected = ci_lower > null_accuracy
    
    if verbose:
        print(f"\n{'='*60}")
        print("Bootstrap Statistical Test Results")
        print(f"{'='*60}")
        print(f"  Number of samples: {n_samples}")
        print(f"  Number of bootstrap resamples: {n_bootstrap}")
        print(f"  Point estimate accuracy: {point_accuracy:.4f}")
        print(f"  {confidence_level*100:.0f}% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  P-value (H0: accuracy = {null_accuracy}): {p_value:.4f}")
        print(f"  Detection confidence: {detection_confidence*100:.1f}%")
        print(f"{'='*60}")
        
        if parity_violation_detected:
            print(f"\n✓ PARITY VIOLATION DETECTED")
            print(f"  We are {detection_confidence*100:.1f}% confident that the statistical")
            print(f"  properties of this field are consistent with a field that violates parity.")
        else:
            print(f"\n✗ NO SIGNIFICANT PARITY VIOLATION DETECTED")
            print(f"  The {confidence_level*100:.0f}% confidence interval includes {null_accuracy}")
            print(f"  Cannot reject null hypothesis of parity symmetry.")
        print(f"{'='*60}\n")
    
    return {
        'accuracy': point_accuracy,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'p_value': p_value,
        'parity_violation_detected': parity_violation_detected,
        'detection_confidence': detection_confidence,
        'bootstrap_accuracies': bootstrap_accuracies
    }


def run_bootstrap_statistical_test(
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1000,
    alpha: float = 0.3,
    f_pv: float = 1.0,
    hidden_dim: int = 16,
    n_layers: int = 2,
    batch_size: int = 64,
    n_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    verbose: bool = True,
    early_stopping_patience: int = None,
    early_stopping_min_delta: float = 1e-4,
    model_type: str = DEFAULT_MODEL_TYPE,
    num_slots: int = 8,
    num_hops: int = 2
):
    """
    Train a model and perform bootstrap statistical test for parity violation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print(f"Using model type: {model_type}")
    
    # Create datasets
    train_dataset = ParityViolationDataset(n_train, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TRAIN)
    val_dataset = ParityViolationDataset(n_val, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_VAL)
    test_dataset = ParityViolationDataset(n_test, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TEST)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_slots=num_slots,
        num_hops=num_hops
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta
    )
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_metrics = evaluate(model, val_loader, criterion, device, model_type)
        
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        if early_stopping(val_metrics['loss'], model):
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {early_stopping_patience} epochs)")
            early_stopping.restore_best_model(model)
            break
        
        if verbose:
            print(f"Epoch {epoch + 1:3d}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_metrics['loss']:.4f}, "
                  f"Val Acc = {val_metrics['accuracy']:.4f}")
    
    # Perform bootstrap statistical test
    if verbose:
        print("\nPerforming bootstrap statistical test on test set...")
    
    bootstrap_seed = seed * SEED_MULTIPLIER + SEED_OFFSET_BOOTSTRAP
    
    bootstrap_results = bootstrap_confidence_test(
        model=model,
        dataloader=test_loader,
        device=device,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        verbose=verbose,
        model_type=model_type
    )
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': bootstrap_results['accuracy'],
        'ci_lower': bootstrap_results['ci_lower'],
        'ci_upper': bootstrap_results['ci_upper'],
        'p_value': bootstrap_results['p_value'],
        'parity_violation_detected': bootstrap_results['parity_violation_detected'],
        'detection_confidence': bootstrap_results['detection_confidence'],
        'epochs_trained': len(train_losses)
    }


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_spin2_orientations(
    positions: np.ndarray,
    angles: np.ndarray,
    labels: np.ndarray = None,
    title: str = "Spin-2 Point Cloud",
    line_length: float = 0.3,
    figsize: tuple = (10, 8),
    cmap: str = 'coolwarm',
    save_path: str = None,
    show_colorbar: bool = True,
    subset_size: int = None,
    seed: int = 42
):
    """Visualize spin-2 orientations as line segments on a 2D point cloud."""
    n_samples = positions.shape[0]
    
    if subset_size is not None and subset_size < n_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_samples, size=subset_size, replace=False)
        positions = positions[indices]
        angles = angles[indices]
        if labels is not None:
            labels = labels[indices]
        n_samples = subset_size
    
    x = positions[:, :, 0].flatten()
    y = positions[:, :, 1].flatten()
    z = positions[:, :, 2].flatten()
    phi = angles.flatten()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    z_min, z_max = z.min(), z.max()
    norm = Normalize(vmin=z_min, vmax=z_max)
    colormap = cm.get_cmap(cmap)
    
    half_len = line_length / 2
    
    segments = []
    colors = []
    
    for i in range(len(x)):
        dx = half_len * np.cos(phi[i])
        dy = half_len * np.sin(phi[i])
        
        x0, y0 = x[i] - dx, y[i] - dy
        x1, y1 = x[i] + dx, y[i] + dy
        
        segments.append([(x0, y0), (x1, y1)])
        colors.append(colormap(norm(z[i])))
    
    lc = LineCollection(segments, colors=colors, linewidths=1.5, alpha=0.8)
    ax.add_collection(lc)
    
    scatter = ax.scatter(x, y, c=z, cmap=cmap, s=15, alpha=0.6, edgecolors='none')
    
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, label='z (line-of-sight)')
        cbar.ax.tick_params(labelsize=10)
    
    for i in range(n_samples):
        p1 = positions[i, 0, :2]
        p2 = positions[i, 1, :2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def plot_pv_dataset(
    n_samples: int = 500,
    alpha: float = 0.3,
    seed: int = 42,
    subset_size: int = 50,
    save_path: str = None,
    **kwargs
):
    """Visualize the parity-violating dataset."""
    dataset = ParityViolationDataset(n_samples=n_samples, alpha=alpha, seed=seed)
    
    n_pv = n_samples // 2
    positions = dataset.positions_np[:n_pv]
    angles = dataset.angles_np[:n_pv]
    labels = dataset.labels[:n_pv].numpy()
    
    return plot_spin2_orientations(
        positions=positions,
        angles=angles,
        labels=labels,
        title=f"Parity-Violating Dataset (α={alpha}, spin-2)\nLine segments show orientation, color shows z-depth",
        subset_size=subset_size,
        seed=seed,
        save_path=save_path,
        **kwargs
    )


def plot_null_dataset(
    n_samples: int = 500,
    seed: int = 42,
    subset_size: int = 50,
    save_path: str = None,
    **kwargs
):
    """Visualize the parity-symmetric (null test) dataset."""
    dataset = ParitySymmetricDataset(n_samples=n_samples, seed=seed)
    
    n_half = n_samples // 2
    positions = dataset.positions_np[:n_half]
    angles = dataset.angles_np[:n_half]
    labels = dataset.labels[:n_half].numpy()
    
    return plot_spin2_orientations(
        positions=positions,
        angles=angles,
        labels=labels,
        title="Null Test Dataset (no parity violation, spin-2)\nLine segments show orientation, color shows z-depth",
        subset_size=subset_size,
        seed=seed,
        save_path=save_path,
        **kwargs
    )


def plot_pv_structure(
    n_samples: int = 200,
    alpha: float = 0.3,
    seed: int = 42,
    save_path: str = None,
    figsize: tuple = (14, 5)
):
    """Visualize the parity-violating structure by showing pairs color-coded by delta_z sign."""
    dataset = ParityViolationDataset(n_samples=n_samples, alpha=alpha, seed=seed)
    
    n_pv = n_samples // 2
    positions = dataset.positions_np[:n_pv]
    angles = dataset.angles_np[:n_pv]
    delta_z = dataset.edge_delta_zs[:n_pv].numpy()
    sin_2delta_phi = dataset.edge_sin_2delta_phis[:n_pv].numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: sin(2Δφ) vs delta_z
    ax1 = axes[0]
    colors = ['red' if dz > 0 else 'blue' for dz in delta_z]
    ax1.scatter(delta_z, sin_2delta_phi, c=colors, alpha=0.5, s=30)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Δz (z₂ - z₁)', fontsize=12)
    ax1.set_ylabel('sin(2Δφ)', fontsize=12)
    ax1.set_title('Parity-Violating Correlation\n(sin(2Δφ) × Δz > 0 expected)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    ax2 = axes[1]
    positive_dz = sin_2delta_phi[delta_z > 0]
    negative_dz = sin_2delta_phi[delta_z < 0]
    ax2.hist(positive_dz, bins=30, alpha=0.6, color='red', label=f'Δz > 0 (n={len(positive_dz)})')
    ax2.hist(negative_dz, bins=30, alpha=0.6, color='blue', label=f'Δz < 0 (n={len(negative_dz)})')
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('sin(2Δφ)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of sin(2Δφ) by Δz sign', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample pairs
    ax3 = axes[2]
    n_show = min(30, n_pv)
    for i in range(n_show):
        p1 = positions[i, 0]
        p2 = positions[i, 1]
        color = 'red' if delta_z[i] > 0 else 'blue'
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, alpha=0.5, linewidth=1)
        
        for j, (pos, ang) in enumerate([(p1, angles[i, 0]), (p2, angles[i, 1])]):
            dx = 0.2 * np.cos(ang)
            dy = 0.2 * np.sin(ang)
            ax3.plot([pos[0]-dx, pos[0]+dx], [pos[1]-dy, pos[1]+dy], 
                    color=color, linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title(f'Sample Pairs (n={n_show})\nRed: Δz>0, Blue: Δz<0', fontsize=11)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, axes


def plot_pv_comparison(
    n_samples: int = 1000,
    alpha: float = 0.3,
    f_pv: float = 1.0,
    seed: int = 42,
    save_path: str = None,
    figsize: tuple = (16, 10)
):
    """
    Visualize parity violation comparison between real and symmetrized data.
    
    This visualization demonstrates that:
    1. Individual pairs look identical between datasets (top row)
    2. Only statistical aggregation reveals the difference (bottom row)
    3. The signal is subtle but detectable through correlation analysis
    
    Args:
        n_samples: Number of samples to generate
        alpha: Parity violation angle offset
        f_pv: Fraction of parity-violating pairs (1.0 = all PV)
        seed: Random seed for reproducibility
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Figure and axes
    """
    # Generate dataset with both real (PV) and symmetrized samples
    dataset = ParityViolationDataset(n_samples=n_samples, alpha=alpha, f_pv=f_pv, seed=seed)
    
    # Split into real (label=1) and symmetrized (label=0) samples
    n_each = n_samples // 2
    
    # Real (parity-violating) samples
    real_positions = dataset.positions_np[:n_each]
    real_angles = dataset.angles_np[:n_each]
    real_delta_z = dataset.edge_delta_zs[:n_each].numpy()
    real_sin_2delta_phi = dataset.edge_sin_2delta_phis[:n_each].numpy()
    
    # Symmetrized samples
    sym_positions = dataset.positions_np[n_each:]
    sym_angles = dataset.angles_np[n_each:]
    sym_delta_z = dataset.edge_delta_zs[n_each:].numpy()
    sym_sin_2delta_phi = dataset.edge_sin_2delta_phis[n_each:].numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # ========================================
    # Top Row: Individual Sample Comparison
    # ========================================
    
    # Panel 1 (top-left): A few sample pairs from REAL data
    ax1 = axes[0, 0]
    n_show = 15
    rng = np.random.default_rng(seed)
    show_indices = rng.choice(n_each, size=n_show, replace=False)
    
    for i in show_indices:
        p1 = real_positions[i, 0]
        p2 = real_positions[i, 1]
        color = 'red' if real_delta_z[i] > 0 else 'blue'
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, alpha=0.4, linewidth=1)
        
        for pos, ang in [(p1, real_angles[i, 0]), (p2, real_angles[i, 1])]:
            dx = 0.25 * np.cos(ang)
            dy = 0.25 * np.sin(ang)
            ax1.plot([pos[0]-dx, pos[0]+dx], [pos[1]-dy, pos[1]+dy], 
                    color=color, linewidth=2.5, alpha=0.9)
    
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('REAL (Parity-Violating) Samples\n(Red: Δz>0, Blue: Δz<0)', fontsize=11)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2 (top-center): A few sample pairs from SYMMETRIZED data
    ax2 = axes[0, 1]
    
    for i in show_indices:
        p1 = sym_positions[i, 0]
        p2 = sym_positions[i, 1]
        color = 'red' if sym_delta_z[i] > 0 else 'blue'
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, alpha=0.4, linewidth=1)
        
        for pos, ang in [(p1, sym_angles[i, 0]), (p2, sym_angles[i, 1])]:
            dx = 0.25 * np.cos(ang)
            dy = 0.25 * np.sin(ang)
            ax2.plot([pos[0]-dx, pos[0]+dx], [pos[1]-dy, pos[1]+dy], 
                    color=color, linewidth=2.5, alpha=0.9)
    
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('SYMMETRIZED Samples\n(Red: Δz>0, Blue: Δz<0)', fontsize=11)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3 (top-right): Explanation text
    ax3 = axes[0, 2]
    ax3.axis('off')
    explanation_text = (
        "Individual samples look IDENTICAL!\n\n"
        "• Both datasets have pairs of points\n"
        "• Both have random positions in 3D\n"
        "• Both have spin-2 orientations\n\n"
        "The difference is STATISTICAL:\n"
        "• Real: orientations correlate with Δz\n"
        "• Symmetrized: 50% are randomly flipped\n\n"
        "This is why detection is non-trivial:\n"
        "A human cannot tell them apart by eye!"
    )
    ax3.text(0.5, 0.5, explanation_text, transform=ax3.transAxes,
             fontsize=12, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    ax3.set_title('Why is detection hard?', fontsize=11)
    
    # ========================================
    # Bottom Row: Statistical Signal (Aggregated)
    # ========================================
    
    # Panel 4 (bottom-left): sin(2Δφ) vs Δz for REAL data
    ax4 = axes[1, 0]
    colors_real = ['red' if dz > 0 else 'blue' for dz in real_delta_z]
    ax4.scatter(real_delta_z, real_sin_2delta_phi, c=colors_real, alpha=0.3, s=15)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.4)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.4)
    
    # Add mean lines to show the correlation (with safety checks for empty arrays)
    pos_mask_real = real_delta_z > 0
    neg_mask_real = real_delta_z < 0
    if np.any(pos_mask_real):
        mean_pos = np.mean(real_sin_2delta_phi[pos_mask_real])
        ax4.axhline(y=mean_pos, color='red', linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Mean (Δz>0): {mean_pos:.3f}')
    if np.any(neg_mask_real):
        mean_neg = np.mean(real_sin_2delta_phi[neg_mask_real])
        ax4.axhline(y=mean_neg, color='blue', linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Mean (Δz<0): {mean_neg:.3f}')
    
    ax4.set_xlabel('Δz (line-of-sight separation)', fontsize=11)
    ax4.set_ylabel('sin(2Δφ)', fontsize=11)
    ax4.set_title('REAL: Clear Correlation!\n(sin(2Δφ) × Δz > 0)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-1.1, 1.1])
    
    # Panel 5 (bottom-center): sin(2Δφ) vs Δz for SYMMETRIZED data
    ax5 = axes[1, 1]
    colors_sym = ['red' if dz > 0 else 'blue' for dz in sym_delta_z]
    ax5.scatter(sym_delta_z, sym_sin_2delta_phi, c=colors_sym, alpha=0.3, s=15)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.4)
    ax5.axvline(x=0, color='k', linestyle='--', alpha=0.4)
    
    # Add mean lines (with safety checks for empty arrays)
    pos_mask_sym = sym_delta_z > 0
    neg_mask_sym = sym_delta_z < 0
    if np.any(pos_mask_sym):
        mean_pos_sym = np.mean(sym_sin_2delta_phi[pos_mask_sym])
        ax5.axhline(y=mean_pos_sym, color='red', linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Mean (Δz>0): {mean_pos_sym:.3f}')
    if np.any(neg_mask_sym):
        mean_neg_sym = np.mean(sym_sin_2delta_phi[neg_mask_sym])
        ax5.axhline(y=mean_neg_sym, color='blue', linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Mean (Δz<0): {mean_neg_sym:.3f}')
    
    ax5.set_xlabel('Δz (line-of-sight separation)', fontsize=11)
    ax5.set_ylabel('sin(2Δφ)', fontsize=11)
    ax5.set_title('SYMMETRIZED: No Correlation!\n(Random around zero)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9, loc='lower right')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([-1.1, 1.1])
    
    # Panel 6 (bottom-right): Histogram comparison
    ax6 = axes[1, 2]
    
    # Compute the signed correlation metric: sign(Δz) × sin(2Δφ)
    real_metric = np.sign(real_delta_z) * real_sin_2delta_phi
    sym_metric = np.sign(sym_delta_z) * sym_sin_2delta_phi
    
    bins = np.linspace(-1, 1, 31)
    ax6.hist(real_metric, bins=bins, alpha=0.6, color='green', 
             label=f'Real (mean={np.mean(real_metric):.3f})', density=True)
    ax6.hist(sym_metric, bins=bins, alpha=0.6, color='gray', 
             label=f'Symmetrized (mean={np.mean(sym_metric):.3f})', density=True)
    
    ax6.axvline(x=np.mean(real_metric), color='green', linestyle='-', linewidth=2)
    ax6.axvline(x=np.mean(sym_metric), color='gray', linestyle='-', linewidth=2)
    ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    ax6.set_xlabel('sign(Δz) × sin(2Δφ)', fontsize=11)
    ax6.set_ylabel('Density', fontsize=11)
    ax6.set_title('The Parity-Odd Statistic\n(Real shifted right, Sym. centered)', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Parity Violation Detection: Real vs Symmetrized Data (α={alpha}, f_pv={f_pv})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, axes


def plot_training_convergence(
    train_losses: list,
    val_losses: list,
    val_accuracies: list = None,
    title: str = "Training Convergence",
    save_path: str = None,
    figsize: tuple = (12, 4)
):
    """Plot training loss convergence over epochs."""
    epochs = range(1, len(train_losses) + 1)
    
    n_plots = 2 if val_accuracies is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (BCE)', fontsize=12)
    ax1.set_title('Loss vs Epoch', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    if val_accuracies is not None:
        ax2 = axes[1]
        ax2.plot(epochs, val_accuracies, 'g-o', label='Val Accuracy', markersize=4)
        ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy vs Epoch', fontsize=12)
        ax2.set_ylim([0.4, 1.0])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, axes


def plot_comparison(
    pv_results: dict,
    control_results: dict,
    save_path: str = None,
    figsize: tuple = (14, 5)
):
    """Compare training convergence between PV and control experiments."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(pv_results['train_losses']) + 1)
    
    ax1 = axes[0]
    ax1.plot(epochs, pv_results['train_losses'], 'b-', label='PV Train', linewidth=2)
    ax1.plot(epochs, control_results['train_losses'], 'r--', label='Control Train', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(epochs, pv_results['val_losses'], 'b-', label='PV Val', linewidth=2)
    ax2.plot(epochs, control_results['val_losses'], 'r--', label='Control Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.plot(epochs, pv_results['val_accuracies'], 'b-', label='PV Accuracy', linewidth=2)
    ax3.plot(epochs, control_results['val_accuracies'], 'r--', label='Control Accuracy', linewidth=2)
    ax3.axhline(y=0.5, color='k', linestyle=':', alpha=0.5, label='Random (0.5)')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation Accuracy', fontsize=12)
    ax3.set_title('Accuracy Comparison', fontsize=12)
    ax3.set_ylim([0.35, 1.0])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('PV Detection vs Null Test Comparison (Spin-2)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, axes


def generate_visualizations(output_dir: str = 'visualizations', verbose: bool = True):
    """Generate all visualizations for the experiment."""
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("Generating visualizations for spin-2 parity violation experiment...")
    
    # 1. Main visualization: Real vs Symmetrized comparison
    if verbose:
        print("\n1. Visualizing parity violation comparison (real vs symmetrized)...")
    plot_pv_comparison(
        n_samples=1000,
        alpha=0.3,
        f_pv=1.0,
        save_path=os.path.join(output_dir, 'pv_comparison.png')
    )
    
    # 2. Visualize PV dataset
    if verbose:
        print("\n2. Visualizing parity-violating dataset...")
    plot_pv_dataset(
        n_samples=500,
        alpha=0.3,
        subset_size=60,
        save_path=os.path.join(output_dir, 'pv_dataset.png')
    )
    
    # 3. Visualize null test dataset
    if verbose:
        print("\n3. Visualizing null test dataset...")
    plot_null_dataset(
        n_samples=500,
        subset_size=60,
        save_path=os.path.join(output_dir, 'null_dataset.png')
    )
    
    # 4. Visualize PV structure
    if verbose:
        print("\n4. Visualizing parity-violating structure...")
    plot_pv_structure(
        n_samples=400,
        alpha=0.3,
        save_path=os.path.join(output_dir, 'pv_structure.png')
    )
    
    plt.close('all')
    
    if verbose:
        print(f"\nAll visualizations saved to '{output_dir}/' directory")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Basic training for 3D parity violation detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--n-train', type=int, default=4000,
                        help='Number of training samples')
    parser.add_argument('--n-val', type=int, default=1000,
                        help='Number of validation samples')
    parser.add_argument('--n-test', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Parity violation parameter (angle offset)')
    parser.add_argument('--f-pv', type=float, default=1.0,
                        help='Fraction of pairs that are parity-violating (0.0 to 1.0)')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default=DEFAULT_MODEL_TYPE,
                        choices=[MODEL_TYPE_EGNN, MODEL_TYPE_FRAME_ALIGNED,
                                 MODEL_TYPE_MULTI_HOP_EGNN, MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED],
                        help='Model type')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for the model')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Number of message passing layers (for egnn)')
    parser.add_argument('--num-slots', type=int, default=32,
                        help='Number of latent slots (for frame_aligned)')
    parser.add_argument('--num-hops', type=int, default=3,
                        help='Number of message passing hops (for frame_aligned)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--n-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Early stopping patience')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-4,
                        help='Minimum improvement for early stopping')
    
    # Statistical test parameters
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of seeds for multi-seed statistical test')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap resamples')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Confidence level for bootstrap intervals')
    
    # Experiment mode
    parser.add_argument('--mode', type=str, default='full',
                        choices=['main', 'control', 'statistical', 'bootstrap', 'full'],
                        help='Experiment mode')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Output directory for visualizations')
    
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    verbose = not args.quiet
    
    print("="*60)
    print("3D Parity-Violating Classifier Experiment (Spin-2)")
    print("="*60)
    
    if verbose:
        print(f"\nConfiguration:")
        print(f"  model_type={args.model_type}")
        print(f"  n_train={args.n_train}, n_val={args.n_val}, n_test={args.n_test}")
        print(f"  alpha={args.alpha}, f_pv={args.f_pv}, hidden_dim={args.hidden_dim}")
        if args.model_type == MODEL_TYPE_EGNN:
            print(f"  n_layers={args.n_layers}")
        else:
            print(f"  num_slots={args.num_slots}, num_hops={args.num_hops}")
        print(f"  batch_size={args.batch_size}, n_epochs={args.n_epochs}, lr={args.lr}")
        print(f"  seed={args.seed}, mode={args.mode}")
    
    # Generate visualizations if requested
    if args.visualize:
        generate_visualizations(output_dir=args.output_dir, verbose=verbose)
    
    results = None
    control_results = None
    stats = None
    bootstrap_results = None
    
    if args.mode in ['main', 'full']:
        print(f"\n[1] Main Experiment (with 3D parity violation, α={args.alpha}, f_pv={args.f_pv})")
        print("-"*60)
        results = run_experiment(
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            alpha=args.alpha,
            f_pv=args.f_pv,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr=args.lr,
            seed=args.seed,
            verbose=verbose,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            model_type=args.model_type,
            num_slots=args.num_slots,
            num_hops=args.num_hops
        )
    
    if args.mode in ['control', 'full']:
        print("\n[2] Control Experiment (no parity violation)")
        print("-"*60)
        control_results = run_control_experiment(
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr=args.lr,
            seed=args.seed,
            verbose=verbose,
            model_type=args.model_type,
            num_slots=args.num_slots,
            num_hops=args.num_hops
        )
    
    if args.mode in ['statistical', 'full']:
        print(f"\n[3] Statistical Test ({args.n_seeds} seeds)")
        print("-"*60)
        stats = run_statistical_test(
            n_seeds=args.n_seeds,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            alpha=args.alpha,
            f_pv=args.f_pv,
            n_epochs=args.n_epochs,
            verbose=verbose,
            model_type=args.model_type,
            num_slots=args.num_slots,
            num_hops=args.num_hops
        )
    
    if args.mode in ['bootstrap', 'full']:
        print(f"\n[4] Bootstrap Statistical Test (n_bootstrap={args.n_bootstrap})")
        print("-"*60)
        bootstrap_results = run_bootstrap_statistical_test(
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            alpha=args.alpha,
            f_pv=args.f_pv,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr=args.lr,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
            confidence_level=args.confidence_level,
            verbose=verbose,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            model_type=args.model_type,
            num_slots=args.num_slots,
            num_hops=args.num_hops
        )
    
    # Generate training comparison visualization if both main and control ran
    if args.visualize and results is not None and control_results is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
        plot_training_convergence(
            train_losses=results['train_losses'],
            val_losses=results['val_losses'],
            val_accuracies=results['val_accuracies'],
            title="PV Detection Training Convergence (Spin-2)",
            save_path=os.path.join(args.output_dir, 'pv_convergence.png')
        )
        
        plot_comparison(
            pv_results=results,
            control_results=control_results,
            save_path=os.path.join(args.output_dir, 'pv_vs_control.png')
        )
        
        plt.close('all')
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if results is not None:
        print(f"Main experiment accuracy: {results['test_accuracy']:.4f}")
    
    if control_results is not None:
        print(f"Control experiment accuracy: {control_results['test_accuracy']:.4f}")
    
    if stats is not None:
        print(f"Statistical test mean accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    
    if bootstrap_results is not None:
        print(f"\nBootstrap Statistical Test Results:")
        print(f"  Test accuracy: {bootstrap_results['test_accuracy']:.4f}")
        print(f"  {args.confidence_level*100:.0f}% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")
        print(f"  P-value: {bootstrap_results['p_value']:.4f}")
        print(f"  Detection confidence: {bootstrap_results['detection_confidence']*100:.1f}%")
        
        if bootstrap_results['parity_violation_detected']:
            print(f"\n✓ We are {bootstrap_results['detection_confidence']*100:.1f}% confident that the")
            print(f"  statistical properties of this field are consistent with")
            print(f"  a field that violates parity.")
        else:
            print(f"\n✗ Cannot conclude parity violation at {args.confidence_level*100:.0f}% confidence level.")
    
    # Validation check
    if args.mode == 'full' and results is not None and control_results is not None:
        print()
        if results['test_accuracy'] > 0.55 and control_results['test_accuracy'] < 0.55:
            print("✓ VALIDATION PASSED: Model correctly detects 3D parity violation")
        else:
            print("⚠ VALIDATION INCONCLUSIVE: Check experiment parameters")

"""
Training module for the parity violation EGNN classifier.

Implements:
- Training loop with BCE loss
- Validation and test evaluation
- Accuracy and loss tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data import ParityViolationDataset, ParitySymmetricDataset
from model import ParityViolationEGNN


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: The EGNN classifier
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function (BCE)
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        node_features = batch['node_features'].to(device)
        edge_distance = batch['edge_distance'].to(device)
        edge_sin_delta_phi = batch['edge_sin_delta_phi'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(node_features, edge_distance, edge_sin_delta_phi)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: The EGNN classifier
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary with loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            node_features = batch['node_features'].to(device)
            edge_distance = batch['edge_distance'].to(device)
            edge_sin_delta_phi = batch['edge_sin_delta_phi'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(node_features, edge_distance, edge_sin_delta_phi)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            
            # Compute accuracy
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def run_experiment(
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1000,
    alpha: float = 0.5,
    hidden_dim: int = 16,
    n_layers: int = 2,
    batch_size: int = 64,
    n_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True
):
    """
    Run a complete training experiment.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        alpha: Parity violation parameter
        hidden_dim: Hidden dimension for the model
        n_layers: Number of message passing layers
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary with final results
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Create datasets (use well-separated seeds for independence)
    train_dataset = ParityViolationDataset(n_train, alpha=alpha, seed=seed * 1000)
    val_dataset = ParityViolationDataset(n_val, alpha=alpha, seed=seed * 1000 + 1)
    test_dataset = ParityViolationDataset(n_test, alpha=alpha, seed=seed * 1000 + 2)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model = ParityViolationEGNN(
        node_input_dim=2,
        edge_input_dim=2,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
        
        if verbose:
            print(f"Epoch {epoch + 1:3d}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_metrics['loss']:.4f}, "
                  f"Val Acc = {val_metrics['accuracy']:.4f}")
    
    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    if verbose:
        print(f"\nTest Results: "
              f"Loss = {test_metrics['loss']:.4f}, "
              f"Accuracy = {test_metrics['accuracy']:.4f}")
    
    return {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'best_val_accuracy': best_val_acc
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
    verbose: bool = True
):
    """
    Run control experiment with parity-symmetric data.
    
    This should yield ~0.5 accuracy (random guessing).
    
    Args:
        Same as run_experiment except no alpha parameter
        
    Returns:
        Dictionary with final results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print("Running CONTROL experiment (no parity violation)")
    
    # Create parity-symmetric datasets (use well-separated seeds for independence)
    train_dataset = ParitySymmetricDataset(n_train, seed=seed * 1000)
    val_dataset = ParitySymmetricDataset(n_val, seed=seed * 1000 + 1)
    test_dataset = ParitySymmetricDataset(n_test, seed=seed * 1000 + 2)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model = ParityViolationEGNN(
        node_input_dim=2,
        edge_input_dim=2,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        if verbose:
            print(f"Epoch {epoch + 1:3d}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_metrics['loss']:.4f}, "
                  f"Val Acc = {val_metrics['accuracy']:.4f}")
    
    # Final evaluation
    test_metrics = evaluate(model, test_loader, criterion, device)
    if verbose:
        print(f"\nControl Test Results: "
              f"Loss = {test_metrics['loss']:.4f}, "
              f"Accuracy = {test_metrics['accuracy']:.4f}")
        print("(Expected: ~0.5 accuracy for parity-symmetric data)")
    
    return {
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy']
    }


def run_statistical_test(
    n_seeds: int = 5,
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1000,
    alpha: float = 0.5,
    n_epochs: int = 20,
    verbose: bool = True
):
    """
    Run multiple experiments with different seeds for statistical robustness.
    
    Args:
        n_seeds: Number of random seeds to try
        n_train: Training samples per experiment
        n_val: Validation samples per experiment
        n_test: Test samples per experiment
        alpha: Parity violation parameter
        n_epochs: Epochs per experiment
        verbose: Print progress
        
    Returns:
        Dictionary with mean and std of test accuracies
    """
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
            n_epochs=n_epochs,
            seed=seed,
            verbose=verbose
        )
        accuracies.append(results['test_accuracy'])
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Statistical Test Results (alpha={alpha}):")
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


if __name__ == '__main__':
    print("="*60)
    print("2D Parity-Violating EGNN Experiment")
    print("="*60)
    
    # Run main experiment with parity violation
    print("\n[1] Main Experiment (with parity violation, α=0.5)")
    print("-"*60)
    results = run_experiment(alpha=0.5, verbose=True)
    
    # Run control experiment without parity violation
    print("\n[2] Control Experiment (no parity violation)")
    print("-"*60)
    control_results = run_control_experiment(verbose=True)
    
    # Run statistical test
    print("\n[3] Statistical Test (5 seeds)")
    print("-"*60)
    stats = run_statistical_test(n_seeds=5, verbose=True)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Main experiment accuracy: {results['test_accuracy']:.4f}")
    print(f"Control experiment accuracy: {control_results['test_accuracy']:.4f}")
    print(f"Statistical test mean accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    
    if results['test_accuracy'] > 0.55 and control_results['test_accuracy'] < 0.55:
        print("\n✓ VALIDATION PASSED: Model correctly detects parity violation")
        print("  - Detects parity when present (α > 0)")
        print("  - Returns ~0.5 accuracy for parity-symmetric control")
    else:
        print("\n⚠ VALIDATION INCONCLUSIVE: Check experiment parameters")

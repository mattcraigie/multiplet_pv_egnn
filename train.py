"""
Training module for the 3D parity violation EGNN classifier with spin-2 objects.

Implements:
- Training loop with BCE loss
- Validation and test evaluation
- Accuracy and loss tracking
- Bootstrap-based statistical test for parity violation detection

Supports two model types:
- 'egnn': Original EGNN-like classifier (from model.py)
- 'frame_aligned': Frame-Aligned GNN classifier (from frame_aligned_model.py)

And multi-hop versions:
- 'multi_hop_egnn': Multi-hop EGNN classifier for variable-size graphs
- 'multi_hop_frame_aligned': Multi-hop Frame-Aligned GNN classifier
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data import ParityViolationDataset, ParitySymmetricDataset, MultiHopParityViolationDataset
from model import ParityViolationEGNN, MultiHopParityViolationEGNN
from frame_aligned_model import FrameAlignedPVClassifier, MultiHopFrameAlignedPVClassifier


# Model type constants
MODEL_TYPE_EGNN = 'egnn'
MODEL_TYPE_FRAME_ALIGNED = 'frame_aligned'
MODEL_TYPE_MULTI_HOP_EGNN = 'multi_hop_egnn'
MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED = 'multi_hop_frame_aligned'
# Default to the new Frame-Aligned model as requested.
# Use --model-type egnn to use the original EGNN model.
DEFAULT_MODEL_TYPE = MODEL_TYPE_FRAME_ALIGNED


# Seed generation constants for dataset independence
# Each dataset (train, val, test) uses a different offset from base seed
SEED_MULTIPLIER = 1000  # Multiplier to separate seed ranges
SEED_OFFSET_TRAIN = 0   # Offset for training dataset
SEED_OFFSET_VAL = 1     # Offset for validation dataset  
SEED_OFFSET_TEST = 2    # Offset for test dataset
SEED_OFFSET_BOOTSTRAP = 3  # Offset for bootstrap resampling


def multi_hop_collate_fn(batch):
    """
    Custom collate function for multi-hop graphs with variable edge counts.
    
    Batches graphs together by concatenating node features and edge indices,
    using a batch vector to track which nodes belong to which graph.
    
    Args:
        batch: List of dictionaries from MultiHopParityViolationDataset
        
    Returns:
        Dictionary with batched tensors
    """
    # Initialize lists
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
        
        # Offset edge indices for batching
        edge_index_list.append(sample['edge_index'] + node_offset)
        
        # Offset special pair indices
        special_pair_list.append(sample['special_pair'] + node_offset)
        
        labels_list.append(sample['label'])
        
        # Batch assignment
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
    """
    Train for one epoch.
    
    Args:
        model: The classifier
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function (BCE)
        device: Device to train on
        model_type: Type of model ('egnn', 'frame_aligned', 'multi_hop_egnn', 'multi_hop_frame_aligned')
        
    Returns:
        Average loss for the epoch
    """
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
    """
    Evaluate the model.
    
    Args:
        model: The classifier
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to evaluate on
        model_type: Type of model ('egnn', 'frame_aligned', 'multi_hop_egnn', 'multi_hop_frame_aligned')
        
    Returns:
        Dictionary with loss and accuracy
    """
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
            
            # Compute accuracy
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


class EarlyStopping:
    """
    Early stopping helper to track validation loss and stop training when loss converges.
    """
    
    def __init__(self, patience: int = None, min_delta: float = 1e-4):
        """
        Initialize early stopping tracker.
        
        Args:
            patience: Number of epochs to wait for improvement. If None, disabled.
            min_delta: Minimum change in validation loss to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
        self.enabled = patience is not None
    
    def __call__(self, val_loss: float, model) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save state from
            
        Returns:
            True if training should stop, False otherwise
        """
        if not self.enabled:
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            # Save best model state
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
    """
    Create a model based on the model type.
    
    Args:
        model_type: 'egnn', 'frame_aligned', 'multi_hop_egnn', or 'multi_hop_frame_aligned'
        hidden_dim: Hidden dimension for the model
        n_layers: Number of message passing layers (for egnn)
        num_slots: Number of latent slots (for frame_aligned)
        num_hops: Number of message passing hops (for frame_aligned)
        readout_dim: Readout dimension (for frame_aligned)
        
    Returns:
        The model instance
    """
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
    """
    Run a complete training experiment.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        alpha: Parity violation parameter (default 0.3 for spin-2)
        f_pv: Fraction of pairs that are parity-violating (0.0 to 1.0).
              f_pv=1.0 means all pairs are PV (original behavior).
              f_pv=0.0 means all pairs have random angles (no signal).
        hidden_dim: Hidden dimension for the model
        n_layers: Number of message passing layers
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        verbose: Whether to print progress
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
                                 If None, early stopping is disabled.
        early_stopping_min_delta: Minimum change in validation loss to qualify as improvement.
        model_type: Type of model ('egnn' or 'frame_aligned')
        num_slots: Number of latent slots (for frame_aligned model)
        num_hops: Number of message passing hops (for frame_aligned model)
        
    Returns:
        Dictionary with final results and loss history
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Create datasets (use well-separated seeds for independence)
    train_dataset = ParityViolationDataset(n_train, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TRAIN)
    val_dataset = ParityViolationDataset(n_val, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_VAL)
    test_dataset = ParityViolationDataset(n_test, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TEST)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model based on model type
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
    
    # Loss history for visualization
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
        
        # Early stopping check
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
    
    # Final evaluation on test set
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


def run_multi_hop_experiment(
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1000,
    n_nodes: int = 10,
    alpha: float = 0.3,
    graph_type: str = 'knn',
    k: int = 3,
    r_max: float = 3.0,
    min_hops: int = 2,
    max_hops: int = 4,
    hidden_dim: int = 32,
    n_layers: int = 3,
    batch_size: int = 32,
    n_epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    model_type: str = MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED,
    num_slots: int = 8,
    num_hops: int = 3
):
    """
    Run a training experiment with multi-hop parity violation dataset.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        n_nodes: Number of nodes per graph
        alpha: Parity violation parameter
        graph_type: 'knn' for k-nearest neighbors, 'radius' for radius graph
        k: Number of neighbors for k-NN graph
        r_max: Maximum radius for radius graph
        min_hops: Minimum hops between special pair
        max_hops: Maximum hops between special pair
        hidden_dim: Hidden dimension for the model
        n_layers: Number of message passing layers (for EGNN)
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        verbose: Whether to print progress
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        early_stopping_min_delta: Minimum change in validation loss to qualify as improvement
        model_type: Type of model ('multi_hop_egnn' or 'multi_hop_frame_aligned')
        num_slots: Number of latent slots (for frame_aligned model)
        num_hops: Number of message passing hops (for frame_aligned model)
        
    Returns:
        Dictionary with final results and loss history
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print(f"Running multi-hop experiment with {n_nodes} nodes per graph")
        print(f"Graph type: {graph_type}, min_hops={min_hops}, max_hops={max_hops}")
    
    # Create datasets
    train_dataset = MultiHopParityViolationDataset(
        n_samples=n_train, n_nodes=n_nodes, alpha=alpha,
        graph_type=graph_type, k=k, r_max=r_max,
        min_hops=min_hops, max_hops=max_hops,
        seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TRAIN
    )
    val_dataset = MultiHopParityViolationDataset(
        n_samples=n_val, n_nodes=n_nodes, alpha=alpha,
        graph_type=graph_type, k=k, r_max=r_max,
        min_hops=min_hops, max_hops=max_hops,
        seed=seed * SEED_MULTIPLIER + SEED_OFFSET_VAL
    )
    test_dataset = MultiHopParityViolationDataset(
        n_samples=n_test, n_nodes=n_nodes, alpha=alpha,
        graph_type=graph_type, k=k, r_max=r_max,
        min_hops=min_hops, max_hops=max_hops,
        seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TEST
    )
    
    if verbose:
        print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multi_hop_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=multi_hop_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=multi_hop_collate_fn)
    
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
        
        # Early stopping check
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
    
    # Final evaluation on test set
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
    """
    Run control experiment with parity-symmetric data.
    
    This should yield ~0.5 accuracy (random guessing).
    
    Args:
        Same as run_experiment except no alpha parameter
        
    Returns:
        Dictionary with final results and loss history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print("Running CONTROL experiment (no parity violation)")
    
    # Create parity-symmetric datasets (use well-separated seeds for independence)
    train_dataset = ParitySymmetricDataset(n_train, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TRAIN)
    val_dataset = ParitySymmetricDataset(n_val, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_VAL)
    test_dataset = ParitySymmetricDataset(n_test, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TEST)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model based on model type
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
    
    # Loss history for visualization
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
    """
    Run multiple experiments with different seeds for statistical robustness.
    
    Args:
        n_seeds: Number of random seeds to try
        n_train: Training samples per experiment
        n_val: Validation samples per experiment
        n_test: Test samples per experiment
        alpha: Parity violation parameter
        f_pv: Fraction of pairs that are parity-violating (0.0 to 1.0)
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
    Perform bootstrap resampling to compute confidence intervals for accuracy
    and determine if parity violation is detected with statistical significance.
    
    This test bootstraps over the predictions on validation/test data to understand
    the uncertainty in our accuracy estimate and determine confidence that the
    field violates parity.
    
    Args:
        model: Trained classifier
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for interval (e.g., 0.95 for 95%)
        null_accuracy: Null hypothesis accuracy (0.5 for no parity violation)
        bootstrap_seed: Random seed for bootstrap resampling (None for random)
        verbose: Whether to print results
        model_type: Type of model ('egnn', 'frame_aligned', 'multi_hop_egnn', 'multi_hop_frame_aligned')
        
    Returns:
        Dictionary with:
        - accuracy: Point estimate of accuracy
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - confidence_level: The confidence level used
        - p_value: P-value for rejecting null hypothesis (accuracy = null_accuracy)
        - parity_violation_detected: Boolean indicating significant detection
        - detection_confidence: Confidence level at which parity violation is detected
    """
    model.eval()
    
    # Collect all predictions and labels
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
    
    # Compute point estimate
    correct = (all_predictions == all_labels).astype(float)
    point_accuracy = np.mean(correct)
    
    # Bootstrap resampling
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_accuracies = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_correct = correct[indices]
        bootstrap_accuracies.append(np.mean(bootstrap_correct))
    
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    
    # Compute confidence interval using percentile method
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_accuracies, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_accuracies, 100 * (1 - alpha / 2))
    
    # Compute p-value for one-sided test (accuracy > null_accuracy)
    # Using bootstrap distribution centered at null
    bootstrap_centered = bootstrap_accuracies - point_accuracy + null_accuracy
    p_value = np.mean(bootstrap_centered >= point_accuracy)
    
    # Determine at what confidence level parity violation is detected
    # (what percentage of bootstrap samples have accuracy > null_accuracy)
    detection_confidence = np.mean(bootstrap_accuracies > null_accuracy)
    
    # Determine if parity violation is detected at specified confidence level
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
    
    This combines training with bootstrap-based confidence interval estimation
    to provide a rigorous statistical statement about parity violation detection.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        alpha: Parity violation parameter
        f_pv: Fraction of pairs that are parity-violating (0.0 to 1.0)
        hidden_dim: Hidden dimension for the model
        n_layers: Number of message passing layers
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for interval
        verbose: Whether to print progress
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
                                 If None, early stopping is disabled.
        early_stopping_min_delta: Minimum change in validation loss to qualify as improvement.
        model_type: Type of model ('egnn' or 'frame_aligned')
        num_slots: Number of latent slots (for frame_aligned model)
        num_hops: Number of message passing hops (for frame_aligned model)
        
    Returns:
        Dictionary with training results and bootstrap test results
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
        print(f"Using model type: {model_type}")
    
    # Create datasets (use well-separated seeds for independence)
    train_dataset = ParityViolationDataset(n_train, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TRAIN)
    val_dataset = ParityViolationDataset(n_val, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_VAL)
    test_dataset = ParityViolationDataset(n_test, alpha=alpha, f_pv=f_pv, seed=seed * SEED_MULTIPLIER + SEED_OFFSET_TEST)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model based on model type
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
    
    # Training loop with early stopping
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping
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
        
        # Early stopping check
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
    
    # Perform bootstrap statistical test on test set
    if verbose:
        print("\nPerforming bootstrap statistical test on test set...")
    
    # Use a derived seed for bootstrap resampling to ensure independence
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train classifier for 3D parity violation detection',
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
                        help='Fraction of pairs that are parity-violating (0.0 to 1.0). '
                             'f_pv=1.0 means all pairs are PV (original behavior). '
                             'f_pv=0.0 means all pairs have random angles (no signal). '
                             'Expected accuracy scales as 0.5 + f_pv/4.')
    
    # Multi-hop data parameters
    parser.add_argument('--n-nodes', type=int, default=10,
                        help='Number of nodes per graph (for multi-hop models)')
    parser.add_argument('--graph-type', type=str, default='knn',
                        choices=['knn', 'radius'],
                        help='Graph construction type: knn or radius')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of neighbors for k-NN graph')
    parser.add_argument('--r-max', type=float, default=3.0,
                        help='Maximum radius for radius graph')
    parser.add_argument('--min-hops', type=int, default=2,
                        help='Minimum hops between special pair')
    parser.add_argument('--max-hops', type=int, default=4,
                        help='Maximum hops between special pair')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default=DEFAULT_MODEL_TYPE,
                        choices=[MODEL_TYPE_EGNN, MODEL_TYPE_FRAME_ALIGNED,
                                 MODEL_TYPE_MULTI_HOP_EGNN, MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED],
                        help=f'Model type: {MODEL_TYPE_EGNN} (original EGNN), '
                             f'{MODEL_TYPE_FRAME_ALIGNED} (Frame-Aligned GNN), '
                             f'{MODEL_TYPE_MULTI_HOP_EGNN} (Multi-hop EGNN), '
                             f'{MODEL_TYPE_MULTI_HOP_FRAME_ALIGNED} (Multi-hop Frame-Aligned)')
    parser.add_argument('--hidden-dim', type=int, default=16,
                        help='Hidden dimension for the model')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Number of message passing layers (for egnn)')
    parser.add_argument('--num-slots', type=int, default=8,
                        help='Number of latent slots (for frame_aligned)')
    parser.add_argument('--num-hops', type=int, default=2,
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
    
    # Early stopping parameters
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Number of epochs to wait for improvement before stopping. '
                             'If not provided, early stopping is disabled.')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-4,
                        help='Minimum change in validation loss to qualify as improvement.')
    
    # Statistical test parameters
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of seeds for multi-seed statistical test')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap resamples for confidence intervals')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Confidence level for bootstrap intervals (e.g., 0.95 for 95 percent)')
    
    # Experiment mode
    parser.add_argument('--mode', type=str, default='full',
                        choices=['main', 'control', 'statistical', 'bootstrap', 'full', 'multi_hop'],
                        help='Experiment mode: main (single PV experiment), '
                             'control (parity-symmetric control), '
                             'statistical (multi-seed test), '
                             'bootstrap (single experiment with bootstrap CI), '
                             'full (all experiments), '
                             'multi_hop (multi-hop experiment)')
    
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
    
    results = None
    control_results = None
    stats = None
    bootstrap_results = None
    
    if args.mode in ['main', 'full']:
        # Run main experiment with parity violation
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
        # Run control experiment without parity violation
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
        # Run statistical test with multiple seeds
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
        # Run bootstrap statistical test
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
    
    if args.mode == 'multi_hop':
        # Run multi-hop experiment
        print(f"\n[5] Multi-Hop Experiment")
        print("-"*60)
        multi_hop_results = run_multi_hop_experiment(
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            n_nodes=args.n_nodes,
            alpha=args.alpha,
            graph_type=args.graph_type,
            k=args.k,
            r_max=args.r_max,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
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
        
        # Print multi-hop summary
        print("\n" + "="*60)
        print("MULTI-HOP EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Test accuracy: {multi_hop_results['test_accuracy']:.4f}")
        print(f"Best val accuracy: {multi_hop_results['best_val_accuracy']:.4f}")
        print(f"Epochs trained: {multi_hop_results['epochs_trained']}")
        
        if multi_hop_results['test_accuracy'] > 0.55:
            print("\n✓ Model shows ability to detect parity violation in multi-hop setting")
        else:
            print("\n⚠ Model performance may need tuning for multi-hop setting")
    
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
    
    # Validation check for full mode
    if args.mode == 'full' and results is not None and control_results is not None:
        print()
        if results['test_accuracy'] > 0.55 and control_results['test_accuracy'] < 0.55:
            print("✓ VALIDATION PASSED: Model correctly detects 3D parity violation")
            print("  - Detects parity when present (α > 0)")
            print("  - Returns ~0.5 accuracy for parity-symmetric control")
        else:
            print("⚠ VALIDATION INCONCLUSIVE: Check experiment parameters")

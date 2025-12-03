"""
Data generation module for 2D Parity-Violating EGNN experiment.

This module generates:
1. Isotropic 2D point pairs with parity-violating angle correlations
2. Symmetrized (parity-balanced) control datasets
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_point_pair(
    box_size: float = 10.0,
    min_separation: float = 1.0,
    max_separation: float = 3.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate a pair of 2D points with isotropic and homogeneous positions.
    
    Args:
        box_size: Size of the square box for sampling centers
        min_separation: Minimum separation distance between points
        max_separation: Maximum separation distance between points
        rng: Random number generator
        
    Returns:
        Array of shape (2, 2) with the two point positions
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample pair center uniformly inside the box
    center = rng.uniform(0, box_size, size=2)
    
    # Sample random axis direction
    theta = rng.uniform(0, 2 * np.pi)
    
    # Sample separation distance
    r = rng.uniform(min_separation, max_separation)
    
    # Place points symmetrically around center along direction theta
    offset = (r / 2) * np.array([np.cos(theta), np.sin(theta)])
    point1 = center + offset
    point2 = center - offset
    
    return np.stack([point1, point2], axis=0)


def generate_angles_parity_violating(
    alpha: float = 0.5,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate parity-violating angle correlations for a 2-point graph.
    
    The angles have a fixed signed relative offset:
    - φ₁ = φ₀ + α
    - φ₂ = φ₀ - α
    
    This creates a parity-violating pattern (Δφ = -2α always has same sign).
    
    Args:
        alpha: The angle offset parameter (default 0.5 rad)
        rng: Random number generator
        
    Returns:
        Array of shape (2,) with the two angles
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample base orientation uniformly
    phi0 = rng.uniform(0, 2 * np.pi)
    
    # Assign angles with fixed offset
    phi1 = (phi0 + alpha) % (2 * np.pi)
    phi2 = (phi0 - alpha) % (2 * np.pi)
    
    return np.array([phi1, phi2])


def symmetrize_angles(angles: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Apply parity symmetrization to angles.
    
    With 50% probability, flip all angles: φ → -φ (mod 2π)
    This removes the parity-violating signature while preserving other statistics.
    
    Args:
        angles: Array of angles to symmetrize
        rng: Random number generator
        
    Returns:
        Symmetrized angles
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if rng.random() < 0.5:
        # Flip angles: φ → -φ (mod 2π)
        return (-angles) % (2 * np.pi)
    else:
        return angles.copy()


def compute_node_features(angles: np.ndarray) -> np.ndarray:
    """
    Compute node features from angles.
    
    Node features: (cos φ, sin φ)
    
    Args:
        angles: Array of shape (n_nodes,) with angles
        
    Returns:
        Array of shape (n_nodes, 2) with node features
    """
    cos_phi = np.cos(angles)
    sin_phi = np.sin(angles)
    return np.stack([cos_phi, sin_phi], axis=-1)


def compute_edge_features(positions: np.ndarray, angles: np.ndarray) -> dict:
    """
    Compute edge features for a 2-point graph.
    
    Edge features:
    - distance: pairwise distance
    - sin_delta_phi: sin(φ₂ - φ₁), the parity-odd feature
    
    Args:
        positions: Array of shape (2, 2) with point positions
        angles: Array of shape (2,) with angles
        
    Returns:
        Dictionary with edge features
    """
    # Compute pairwise distance
    distance = np.linalg.norm(positions[1] - positions[0])
    
    # Compute sin(Δφ) = sin(φ₂ - φ₁)
    delta_phi = angles[1] - angles[0]
    sin_delta_phi = np.sin(delta_phi)
    
    return {
        'distance': distance,
        'sin_delta_phi': sin_delta_phi
    }


class ParityViolationDataset(Dataset):
    """
    PyTorch Dataset for parity violation detection.
    
    Each sample contains:
    - node_features: (n_nodes, 2) tensor with (cos φ, sin φ)
    - edge_distance: scalar distance between nodes
    - edge_sin_delta_phi: sin(Δφ) parity-odd feature
    - label: 1 for real (parity-violating), 0 for symmetrized
    """
    
    def __init__(
        self,
        n_samples: int,
        alpha: float = 0.5,
        box_size: float = 10.0,
        min_separation: float = 1.0,
        max_separation: float = 3.0,
        seed: int = None
    ):
        """
        Initialize the dataset.
        
        Args:
            n_samples: Total number of samples (half real, half symmetrized)
            alpha: Parity violation angle offset
            box_size: Box size for position sampling
            min_separation: Minimum point separation
            max_separation: Maximum point separation
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.alpha = alpha
        self.box_size = box_size
        self.min_separation = min_separation
        self.max_separation = max_separation
        
        # Generate all data
        self.rng = np.random.default_rng(seed)
        self._generate_data()
    
    def _generate_data(self):
        """Generate all samples."""
        n_each = self.n_samples // 2
        
        self.node_features = []
        self.edge_distances = []
        self.edge_sin_delta_phis = []
        self.labels = []
        
        # Generate real (parity-violating) samples
        for _ in range(n_each):
            positions = generate_point_pair(
                self.box_size, self.min_separation, self.max_separation, self.rng
            )
            angles = generate_angles_parity_violating(self.alpha, self.rng)
            
            node_feat = compute_node_features(angles)
            edge_feat = compute_edge_features(positions, angles)
            
            self.node_features.append(node_feat)
            self.edge_distances.append(edge_feat['distance'])
            self.edge_sin_delta_phis.append(edge_feat['sin_delta_phi'])
            self.labels.append(1)  # Real sample
        
        # Generate symmetrized samples
        for _ in range(n_each):
            positions = generate_point_pair(
                self.box_size, self.min_separation, self.max_separation, self.rng
            )
            angles = generate_angles_parity_violating(self.alpha, self.rng)
            angles = symmetrize_angles(angles, self.rng)
            
            node_feat = compute_node_features(angles)
            edge_feat = compute_edge_features(positions, angles)
            
            self.node_features.append(node_feat)
            self.edge_distances.append(edge_feat['distance'])
            self.edge_sin_delta_phis.append(edge_feat['sin_delta_phi'])
            self.labels.append(0)  # Symmetrized sample
        
        # Convert to tensors
        self.node_features = torch.tensor(
            np.array(self.node_features), dtype=torch.float32
        )
        self.edge_distances = torch.tensor(
            np.array(self.edge_distances), dtype=torch.float32
        )
        self.edge_sin_delta_phis = torch.tensor(
            np.array(self.edge_sin_delta_phis), dtype=torch.float32
        )
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'node_features': self.node_features[idx],
            'edge_distance': self.edge_distances[idx],
            'edge_sin_delta_phi': self.edge_sin_delta_phis[idx],
            'label': self.labels[idx]
        }


class ParitySymmetricDataset(Dataset):
    """
    Control dataset with completely random angles (no parity violation).
    
    Used to verify that the classifier returns ~0.5 accuracy when there's
    no parity-violating signal.
    """
    
    def __init__(
        self,
        n_samples: int,
        box_size: float = 10.0,
        min_separation: float = 1.0,
        max_separation: float = 3.0,
        seed: int = None
    ):
        """
        Initialize the symmetric control dataset.
        
        Args:
            n_samples: Total number of samples
            box_size: Box size for position sampling
            min_separation: Minimum point separation
            max_separation: Maximum point separation
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.box_size = box_size
        self.min_separation = min_separation
        self.max_separation = max_separation
        
        self.rng = np.random.default_rng(seed)
        self._generate_data()
    
    def _generate_data(self):
        """Generate all samples with random angles."""
        n_each = self.n_samples // 2
        
        self.node_features = []
        self.edge_distances = []
        self.edge_sin_delta_phis = []
        self.labels = []
        
        # Generate samples with random labels and random angles
        for label in [1, 0]:
            for _ in range(n_each):
                positions = generate_point_pair(
                    self.box_size, self.min_separation, self.max_separation, self.rng
                )
                # Completely random angles (no parity violation)
                angles = self.rng.uniform(0, 2 * np.pi, size=2)
                
                node_feat = compute_node_features(angles)
                edge_feat = compute_edge_features(positions, angles)
                
                self.node_features.append(node_feat)
                self.edge_distances.append(edge_feat['distance'])
                self.edge_sin_delta_phis.append(edge_feat['sin_delta_phi'])
                self.labels.append(label)
        
        # Convert to tensors
        self.node_features = torch.tensor(
            np.array(self.node_features), dtype=torch.float32
        )
        self.edge_distances = torch.tensor(
            np.array(self.edge_distances), dtype=torch.float32
        )
        self.edge_sin_delta_phis = torch.tensor(
            np.array(self.edge_sin_delta_phis), dtype=torch.float32
        )
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'node_features': self.node_features[idx],
            'edge_distance': self.edge_distances[idx],
            'edge_sin_delta_phi': self.edge_sin_delta_phis[idx],
            'label': self.labels[idx]
        }

"""
Data generation module for 3D Parity-Violating EGNN experiment.

This module generates:
1. Isotropic 3D point pairs with parity-violating angle correlations
   that depend on the line-of-sight (z) coordinate
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


def generate_point_pair_3d(
    box_size: float = 10.0,
    box_size_z: float = None,
    min_separation: float = 1.0,
    max_separation: float = 3.0,
    dz_max: float = None,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate a pair of 3D points with isotropic and homogeneous positions.
    
    The x, y coordinates are generated as in the 2D case.
    The z coordinate (line-of-sight) is generated symmetrically around a center.
    
    Args:
        box_size: Size of the square box for sampling centers in x, y
        box_size_z: Size of the box for sampling centers in z (defaults to box_size)
        min_separation: Minimum separation distance between points in x, y plane
        max_separation: Maximum separation distance between points in x, y plane
        dz_max: Maximum line-of-sight separation (defaults to max_separation)
        rng: Random number generator
        
    Returns:
        Array of shape (2, 3) with the two point positions [x, y, z]
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if box_size_z is None:
        box_size_z = box_size
    
    if dz_max is None:
        dz_max = max_separation
    
    # Sample pair center uniformly inside the box (x, y)
    center_xy = rng.uniform(0, box_size, size=2)
    
    # Sample random axis direction in x-y plane
    theta = rng.uniform(0, 2 * np.pi)
    
    # Sample separation distance in x-y plane
    r = rng.uniform(min_separation, max_separation)
    
    # Place points symmetrically around center along direction theta
    offset_xy = (r / 2) * np.array([np.cos(theta), np.sin(theta)])
    point1_xy = center_xy + offset_xy
    point2_xy = center_xy - offset_xy
    
    # Sample z-center uniformly
    z_center = rng.uniform(0, box_size_z)
    
    # Sample line-of-sight separation symmetrically
    delta_z = rng.uniform(-dz_max, dz_max)
    
    # Place points symmetrically in z
    z1 = z_center + delta_z / 2
    z2 = z_center - delta_z / 2
    
    # Construct 3D positions
    point1 = np.array([point1_xy[0], point1_xy[1], z1])
    point2 = np.array([point2_xy[0], point2_xy[1], z2])
    
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


def generate_angles_parity_violating_3d(
    alpha: float = 0.5,
    delta_z: float = 0.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate 3D parity-violating angle correlations for a 2-point graph.
    
    The sign of the angle difference Δφ depends on the sign of delta_z:
    - If delta_z > 0: Δφ = φ₂ - φ₁ ≈ +2α
    - If delta_z < 0: Δφ = φ₂ - φ₁ ≈ -2α
    - If delta_z = 0: treated as delta_z > 0
    
    This creates a true 3D parity-violating signal that correlates with 
    line-of-sight ordering.
    
    Args:
        alpha: The angle offset parameter (default 0.5 rad)
        delta_z: The line-of-sight separation (z2 - z1)
        rng: Random number generator
        
    Returns:
        Array of shape (2,) with the two angles
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample base orientation uniformly
    phi0 = rng.uniform(0, 2 * np.pi)
    
    # Assign angles based on sign of delta_z
    if delta_z >= 0:
        # Δφ = phi2 - phi1 ≈ +2α
        phi1 = (phi0 - alpha) % (2 * np.pi)
        phi2 = (phi0 + alpha) % (2 * np.pi)
    else:
        # Δφ = phi2 - phi1 ≈ -2α
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
    - distance_3d: 3D pairwise distance (works for 2D positions as well)
    - delta_z: signed line-of-sight separation (z2 - z1), 0.0 for 2D positions
    - sin_delta_phi: sin(φ₂ - φ₁), the parity-odd feature
    
    Args:
        positions: Array of shape (2, 2) or (2, 3) with point positions
        angles: Array of shape (2,) with angles
        
    Returns:
        Dictionary with edge features
    """
    # Compute pairwise 3D distance (works for any dimension)
    distance_3d = np.linalg.norm(positions[1] - positions[0])
    
    # Compute delta_z (line-of-sight separation)
    if positions.shape[1] >= 3:
        delta_z = positions[1, 2] - positions[0, 2]
    else:
        delta_z = 0.0
    
    # Compute sin(Δφ) = sin(φ₂ - φ₁)
    delta_phi = angles[1] - angles[0]
    sin_delta_phi = np.sin(delta_phi)
    
    return {
        'distance_3d': distance_3d,
        'delta_z': delta_z,
        'sin_delta_phi': sin_delta_phi
    }


class ParityViolationDataset(Dataset):
    """
    PyTorch Dataset for 3D parity violation detection.
    
    Each sample contains:
    - node_features: (n_nodes, 2) tensor with (cos φ, sin φ)
    - edge_distance_3d: scalar 3D distance between nodes
    - edge_delta_z: signed line-of-sight separation (z2 - z1)
    - edge_sin_delta_phi: sin(Δφ) parity-odd feature
    - label: 1 for real (parity-violating), 0 for symmetrized
    """
    
    def __init__(
        self,
        n_samples: int,
        alpha: float = 0.5,
        box_size: float = 10.0,
        box_size_z: float = None,
        min_separation: float = 1.0,
        max_separation: float = 3.0,
        dz_max: float = None,
        seed: int = None
    ):
        """
        Initialize the dataset.
        
        Args:
            n_samples: Total number of samples (half real, half symmetrized)
            alpha: Parity violation angle offset
            box_size: Box size for position sampling in x, y
            box_size_z: Box size for position sampling in z (defaults to box_size)
            min_separation: Minimum point separation in x-y plane
            max_separation: Maximum point separation in x-y plane
            dz_max: Maximum line-of-sight separation (defaults to max_separation)
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.alpha = alpha
        self.box_size = box_size
        self.box_size_z = box_size_z if box_size_z is not None else box_size
        self.min_separation = min_separation
        self.max_separation = max_separation
        self.dz_max = dz_max if dz_max is not None else max_separation
        
        # Generate all data
        self.rng = np.random.default_rng(seed)
        self._generate_data()
    
    def _generate_data(self):
        """Generate all samples."""
        n_each = self.n_samples // 2
        
        self.node_features = []
        self.edge_distances_3d = []
        self.edge_delta_zs = []
        self.edge_sin_delta_phis = []
        self.labels = []
        
        # Generate real (parity-violating) samples
        for _ in range(n_each):
            positions = generate_point_pair_3d(
                self.box_size, self.box_size_z,
                self.min_separation, self.max_separation,
                self.dz_max, self.rng
            )
            # Compute delta_z first (z2 - z1)
            delta_z = positions[1, 2] - positions[0, 2]
            
            # Generate angles with 3D parity-violating rule
            angles = generate_angles_parity_violating_3d(self.alpha, delta_z, self.rng)
            
            node_feat = compute_node_features(angles)
            edge_feat = compute_edge_features(positions, angles)
            
            self.node_features.append(node_feat)
            self.edge_distances_3d.append(edge_feat['distance_3d'])
            self.edge_delta_zs.append(edge_feat['delta_z'])
            self.edge_sin_delta_phis.append(edge_feat['sin_delta_phi'])
            self.labels.append(1)  # Real sample
        
        # Generate symmetrized samples
        for _ in range(n_each):
            positions = generate_point_pair_3d(
                self.box_size, self.box_size_z,
                self.min_separation, self.max_separation,
                self.dz_max, self.rng
            )
            # Compute delta_z first (z2 - z1)
            delta_z = positions[1, 2] - positions[0, 2]
            
            # Generate angles with 3D parity-violating rule, then symmetrize
            angles = generate_angles_parity_violating_3d(self.alpha, delta_z, self.rng)
            angles = symmetrize_angles(angles, self.rng)
            
            node_feat = compute_node_features(angles)
            edge_feat = compute_edge_features(positions, angles)
            
            self.node_features.append(node_feat)
            self.edge_distances_3d.append(edge_feat['distance_3d'])
            self.edge_delta_zs.append(edge_feat['delta_z'])
            self.edge_sin_delta_phis.append(edge_feat['sin_delta_phi'])
            self.labels.append(0)  # Symmetrized sample
        
        # Convert to tensors
        self.node_features = torch.tensor(
            np.array(self.node_features), dtype=torch.float32
        )
        self.edge_distances_3d = torch.tensor(
            np.array(self.edge_distances_3d), dtype=torch.float32
        )
        self.edge_delta_zs = torch.tensor(
            np.array(self.edge_delta_zs), dtype=torch.float32
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
            'edge_distance_3d': self.edge_distances_3d[idx],
            'edge_delta_z': self.edge_delta_zs[idx],
            'edge_sin_delta_phi': self.edge_sin_delta_phis[idx],
            'label': self.labels[idx]
        }


class ParitySymmetricDataset(Dataset):
    """
    Control dataset with completely random angles (no parity violation).
    
    Uses 3D positions but generates completely random angles independent
    of positions and delta_z. Used to verify that the classifier returns
    ~0.5 accuracy when there's no parity-violating signal.
    """
    
    def __init__(
        self,
        n_samples: int,
        box_size: float = 10.0,
        box_size_z: float = None,
        min_separation: float = 1.0,
        max_separation: float = 3.0,
        dz_max: float = None,
        seed: int = None
    ):
        """
        Initialize the symmetric control dataset.
        
        Args:
            n_samples: Total number of samples
            box_size: Box size for position sampling in x, y
            box_size_z: Box size for position sampling in z (defaults to box_size)
            min_separation: Minimum point separation in x-y plane
            max_separation: Maximum point separation in x-y plane
            dz_max: Maximum line-of-sight separation (defaults to max_separation)
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.box_size = box_size
        self.box_size_z = box_size_z if box_size_z is not None else box_size
        self.min_separation = min_separation
        self.max_separation = max_separation
        self.dz_max = dz_max if dz_max is not None else max_separation
        
        self.rng = np.random.default_rng(seed)
        self._generate_data()
    
    def _generate_data(self):
        """Generate all samples with random angles."""
        n_each = self.n_samples // 2
        
        self.node_features = []
        self.edge_distances_3d = []
        self.edge_delta_zs = []
        self.edge_sin_delta_phis = []
        self.labels = []
        
        # Generate samples with random labels and random angles
        for label in [1, 0]:
            for _ in range(n_each):
                positions = generate_point_pair_3d(
                    self.box_size, self.box_size_z,
                    self.min_separation, self.max_separation,
                    self.dz_max, self.rng
                )
                # Completely random angles (no parity violation)
                angles = self.rng.uniform(0, 2 * np.pi, size=2)
                
                node_feat = compute_node_features(angles)
                edge_feat = compute_edge_features(positions, angles)
                
                self.node_features.append(node_feat)
                self.edge_distances_3d.append(edge_feat['distance_3d'])
                self.edge_delta_zs.append(edge_feat['delta_z'])
                self.edge_sin_delta_phis.append(edge_feat['sin_delta_phi'])
                self.labels.append(label)
        
        # Convert to tensors
        self.node_features = torch.tensor(
            np.array(self.node_features), dtype=torch.float32
        )
        self.edge_distances_3d = torch.tensor(
            np.array(self.edge_distances_3d), dtype=torch.float32
        )
        self.edge_delta_zs = torch.tensor(
            np.array(self.edge_delta_zs), dtype=torch.float32
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
            'edge_distance_3d': self.edge_distances_3d[idx],
            'edge_delta_z': self.edge_delta_zs[idx],
            'edge_sin_delta_phi': self.edge_sin_delta_phis[idx],
            'label': self.labels[idx]
        }

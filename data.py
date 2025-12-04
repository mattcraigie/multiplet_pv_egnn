"""
Data generation module for 3D Parity-Violating EGNN experiment with spin-2 objects.

This module generates:
1. Isotropic 3D point pairs with parity-violating angle correlations
   that depend on the line-of-sight (z) coordinate
2. Symmetrized (parity-balanced) control datasets

Spin-2 objects have headless orientations (like line segments) where angle φ
and angle φ+π represent the same orientation. The natural representation uses
cos(2φ), sin(2φ) which have period π, and angles are sampled from [0, π).
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
    alpha: float = 0.3,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate parity-violating angle correlations for a 2-point graph (spin-2, 2D version).
    
    This is the simpler 2D version that creates a fixed-sign parity violation.
    For the 3D version that correlates with line-of-sight separation, see
    generate_angles_parity_violating_3d().
    
    For spin-2 objects, angles are in [0, π) and represent headless orientations.
    The angles have a fixed signed relative offset:
    - φ₁ = φ₀ + α
    - φ₂ = φ₀ - α
    
    This creates a parity-violating pattern (Δφ = -2α always has same sign).
    
    Args:
        alpha: The angle offset parameter (default 0.3 rad, appropriate for spin-2)
        rng: Random number generator
        
    Returns:
        Array of shape (2,) with the two angles in [0, π)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample base orientation uniformly in [0, π) for spin-2
    phi0 = rng.uniform(0, np.pi)
    
    # Assign angles with fixed offset (mod π for spin-2)
    phi1 = (phi0 + alpha) % np.pi
    phi2 = (phi0 - alpha) % np.pi
    
    return np.array([phi1, phi2])


def generate_angles_parity_violating_3d(
    alpha: float = 0.3,
    delta_z: float = 0.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate 3D parity-violating angle correlations for a 2-point graph (spin-2).
    
    For spin-2 objects, angles are in [0, π) and represent headless orientations.
    The sign of the angle difference Δφ depends on the sign of delta_z:
    - If delta_z > 0: Δφ = φ₂ - φ₁ ≈ +2α
    - If delta_z < 0: Δφ = φ₂ - φ₁ ≈ -2α
    - If delta_z = 0 exactly: treated as positive (same as delta_z > 0), 
      though this is rare in practice since delta_z is sampled from a 
      continuous distribution.
    
    This creates a true 3D parity-violating signal that correlates with 
    line-of-sight ordering.
    
    Args:
        alpha: The angle offset parameter (default 0.3 rad, appropriate for spin-2)
        delta_z: The line-of-sight separation (z2 - z1)
        rng: Random number generator
        
    Returns:
        Array of shape (2,) with the two angles in [0, π)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample base orientation uniformly in [0, π) for spin-2
    phi0 = rng.uniform(0, np.pi)
    
    # Assign angles based on sign of delta_z (delta_z == 0 treated as positive)
    # Using mod π for spin-2
    if delta_z >= 0:
        # Δφ = phi2 - phi1 ≈ +2α
        phi1 = (phi0 - alpha) % np.pi
        phi2 = (phi0 + alpha) % np.pi
    else:
        # Δφ = phi2 - phi1 ≈ -2α
        phi1 = (phi0 + alpha) % np.pi
        phi2 = (phi0 - alpha) % np.pi
    
    return np.array([phi1, phi2])


def generate_angles_random(rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate completely random angles for a 2-point graph (spin-2).
    
    For spin-2 objects, angles are sampled uniformly in [0, π).
    This creates parity-symmetric random pairs with no systematic signal.
    
    Args:
        rng: Random number generator
        
    Returns:
        Array of shape (2,) with two random angles in [0, π)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    return rng.uniform(0, np.pi, size=2)


def symmetrize_angles(angles: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Apply parity symmetrization to angles (spin-2).
    
    For spin-2 objects, parity flips the orientation: φ → -φ (mod π).
    With 50% probability, flip all angles to remove the parity-violating signature.
    
    Args:
        angles: Array of angles to symmetrize (in [0, π))
        rng: Random number generator
        
    Returns:
        Symmetrized angles (in [0, π))
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if rng.random() < 0.5:
        # Flip angles: φ → -φ (mod π) for spin-2
        return (-angles) % np.pi
    else:
        return angles.copy()


def compute_node_features(angles: np.ndarray) -> np.ndarray:
    """
    Compute node features from angles (spin-2).
    
    For spin-2 objects, use doubled angles to get period-π features:
    Node features: (cos(2φ), sin(2φ))
    
    Args:
        angles: Array of shape (n_nodes,) with angles in [0, π)
        
    Returns:
        Array of shape (n_nodes, 2) with node features
    """
    # Use 2φ for spin-2 representation (period π)
    cos_2phi = np.cos(2 * angles)
    sin_2phi = np.sin(2 * angles)
    return np.stack([cos_2phi, sin_2phi], axis=-1)


def compute_delta_z(positions: np.ndarray) -> float:
    """
    Compute the line-of-sight separation (z2 - z1) from 3D positions.
    
    Args:
        positions: Array of shape (2, 3) with 3D point positions
        
    Returns:
        Signed line-of-sight separation delta_z = z2 - z1
    """
    return positions[1, 2] - positions[0, 2]


def compute_edge_features(positions: np.ndarray, angles: np.ndarray) -> dict:
    """
    Compute edge features for a 2-point graph (spin-2).
    
    Edge features:
    - distance_3d: 3D pairwise distance (works for 2D positions as well)
    - delta_z: signed line-of-sight separation (z2 - z1), 0.0 for 2D positions
    - sin_2delta_phi: sin(2(φ₂ - φ₁)), the parity-odd feature for spin-2
    
    Args:
        positions: Array of shape (2, 2) or (2, 3) with point positions
        angles: Array of shape (2,) with angles in [0, π)
        
    Returns:
        Dictionary with edge features
    """
    # Compute pairwise 3D distance (works for any dimension)
    distance_3d = np.linalg.norm(positions[1] - positions[0])
    
    # Compute delta_z (line-of-sight separation) using helper for 3D positions
    if positions.shape[1] >= 3:
        delta_z = compute_delta_z(positions)
    else:
        delta_z = 0.0
    
    # Compute sin(2Δφ) = sin(2(φ₂ - φ₁)) for spin-2
    delta_phi = angles[1] - angles[0]
    sin_2delta_phi = np.sin(2 * delta_phi)
    
    return {
        'distance_3d': distance_3d,
        'delta_z': delta_z,
        'sin_2delta_phi': sin_2delta_phi
    }


class ParityViolationDataset(Dataset):
    """
    PyTorch Dataset for 3D parity violation detection with spin-2 objects.
    
    Each sample contains:
    - node_features: (n_nodes, 2) tensor with (cos(2φ), sin(2φ)) for spin-2
    - edge_distance_3d: scalar 3D distance between nodes
    - edge_delta_z: signed line-of-sight separation (z2 - z1)
    - edge_sin_2delta_phi: sin(2Δφ) parity-odd feature for spin-2
    - label: 1 for real (parity-violating), 0 for symmetrized
    
    The f_pv parameter controls what fraction of pairs have parity-violating
    angle correlations vs completely random angles:
    - f_pv = 1.0: All pairs are parity-violating (original behavior)
    - f_pv = 0.0: All pairs have random angles (no signal)
    - Intermediate values interpolate between these extremes
    
    Expected classifier accuracy scales with f_pv:
    - f_pv = 0: ~50% (random guessing)
    - f_pv = 1: ~75% (Bayes optimal)
    - accuracy ≈ 0.5 + f_pv / 4 (theoretical limit)
    """
    
    def __init__(
        self,
        n_samples: int,
        alpha: float = 0.3,
        f_pv: float = 1.0,
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
            alpha: Parity violation angle offset (default 0.3 rad for spin-2)
            f_pv: Fraction of pairs that are parity-violating (0.0 to 1.0).
                  f_pv=1.0 means all pairs are PV (original behavior).
                  f_pv=0.0 means all pairs have random angles (no signal).
            box_size: Box size for position sampling in x, y
            box_size_z: Box size for position sampling in z (defaults to box_size)
            min_separation: Minimum point separation in x-y plane
            max_separation: Maximum point separation in x-y plane
            dz_max: Maximum line-of-sight separation (defaults to max_separation)
            seed: Random seed for reproducibility
        """
        if not 0.0 <= f_pv <= 1.0:
            raise ValueError(f"f_pv must be in [0, 1], got {f_pv}")
        
        self.n_samples = n_samples
        self.alpha = alpha
        self.f_pv = f_pv
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
        self.positions = []  # Store positions for visualization
        self.angles = []  # Store angles for visualization
        self.edge_distances_3d = []
        self.edge_delta_zs = []
        self.edge_sin_2delta_phis = []
        self.labels = []
        
        # Generate real (parity-violating) samples
        # With probability f_pv: use parity-violating angles
        # With probability (1 - f_pv): use completely random angles
        for _ in range(n_each):
            positions = generate_point_pair_3d(
                self.box_size, self.box_size_z,
                self.min_separation, self.max_separation,
                self.dz_max, self.rng
            )
            # Compute delta_z for angle generation
            delta_z = compute_delta_z(positions)
            
            # Generate angles based on f_pv mixture
            if self.rng.random() < self.f_pv:
                # Parity-violating angles
                angles = generate_angles_parity_violating_3d(self.alpha, delta_z, self.rng)
            else:
                # Completely random angles (no parity signal)
                angles = generate_angles_random(self.rng)
            
            node_feat = compute_node_features(angles)
            edge_feat = compute_edge_features(positions, angles)
            
            self.node_features.append(node_feat)
            self.positions.append(positions)
            self.angles.append(angles)
            self.edge_distances_3d.append(edge_feat['distance_3d'])
            self.edge_delta_zs.append(edge_feat['delta_z'])
            self.edge_sin_2delta_phis.append(edge_feat['sin_2delta_phi'])
            self.labels.append(1)  # Real sample
        
        # Generate symmetrized samples
        # Same mixture rule as real samples, then apply symmetrization
        for _ in range(n_each):
            positions = generate_point_pair_3d(
                self.box_size, self.box_size_z,
                self.min_separation, self.max_separation,
                self.dz_max, self.rng
            )
            # Compute delta_z for angle generation
            delta_z = compute_delta_z(positions)
            
            # Generate angles using the same mixture as real samples
            if self.rng.random() < self.f_pv:
                # Parity-violating angles
                angles = generate_angles_parity_violating_3d(self.alpha, delta_z, self.rng)
            else:
                # Completely random angles (no parity signal)
                angles = generate_angles_random(self.rng)
            
            # Apply symmetrization (50% chance to flip angles)
            # Note: For random angles, this flip doesn't change the distribution
            # For PV angles, half are flipped into their mirror version
            angles = symmetrize_angles(angles, self.rng)
            
            node_feat = compute_node_features(angles)
            edge_feat = compute_edge_features(positions, angles)
            
            self.node_features.append(node_feat)
            self.positions.append(positions)
            self.angles.append(angles)
            self.edge_distances_3d.append(edge_feat['distance_3d'])
            self.edge_delta_zs.append(edge_feat['delta_z'])
            self.edge_sin_2delta_phis.append(edge_feat['sin_2delta_phi'])
            self.labels.append(0)  # Symmetrized sample
        
        # Store numpy arrays for visualization
        self.positions_np = np.array(self.positions)
        self.angles_np = np.array(self.angles)
        
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
        self.edge_sin_2delta_phis = torch.tensor(
            np.array(self.edge_sin_2delta_phis), dtype=torch.float32
        )
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'node_features': self.node_features[idx],
            'edge_distance_3d': self.edge_distances_3d[idx],
            'edge_delta_z': self.edge_delta_zs[idx],
            'edge_sin_2delta_phi': self.edge_sin_2delta_phis[idx],
            'label': self.labels[idx]
        }


class ParitySymmetricDataset(Dataset):
    """
    Control dataset with completely random angles (no parity violation).
    
    Uses 3D positions but generates completely random angles (spin-2) independent
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
        """Generate all samples with random angles (spin-2)."""
        n_each = self.n_samples // 2
        
        self.node_features = []
        self.positions = []  # Store positions for visualization
        self.angles = []  # Store angles for visualization
        self.edge_distances_3d = []
        self.edge_delta_zs = []
        self.edge_sin_2delta_phis = []
        self.labels = []
        
        # Generate samples with random labels and random angles
        for label in [1, 0]:
            for _ in range(n_each):
                positions = generate_point_pair_3d(
                    self.box_size, self.box_size_z,
                    self.min_separation, self.max_separation,
                    self.dz_max, self.rng
                )
                # Completely random angles in [0, π) for spin-2 (no parity violation)
                angles = self.rng.uniform(0, np.pi, size=2)
                
                node_feat = compute_node_features(angles)
                edge_feat = compute_edge_features(positions, angles)
                
                self.node_features.append(node_feat)
                self.positions.append(positions)
                self.angles.append(angles)
                self.edge_distances_3d.append(edge_feat['distance_3d'])
                self.edge_delta_zs.append(edge_feat['delta_z'])
                self.edge_sin_2delta_phis.append(edge_feat['sin_2delta_phi'])
                self.labels.append(label)
        
        # Store numpy arrays for visualization
        self.positions_np = np.array(self.positions)
        self.angles_np = np.array(self.angles)
        
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
        self.edge_sin_2delta_phis = torch.tensor(
            np.array(self.edge_sin_2delta_phis), dtype=torch.float32
        )
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'node_features': self.node_features[idx],
            'edge_distance_3d': self.edge_distances_3d[idx],
            'edge_delta_z': self.edge_delta_zs[idx],
            'edge_sin_2delta_phi': self.edge_sin_2delta_phis[idx],
            'label': self.labels[idx]
        }

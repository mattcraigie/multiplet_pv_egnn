"""
Visualization module for the spin-2 parity-violating EGNN experiment.

Provides visualizations for:
1. PV dataset: 2D point cloud with spin-2 orientations as line segments, z-axis as color
2. Non-PV null test dataset: Same visualization for comparison
3. Training loss convergence over epochs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from data import ParityViolationDataset, ParitySymmetricDataset


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
    """
    Visualize spin-2 orientations as line segments on a 2D point cloud.
    
    Each point has a headless orientation (spin-2) represented by a small line segment.
    The z-coordinate (line-of-sight) is indicated by color.
    
    Args:
        positions: Array of shape (n_samples, 2, 3) with 3D positions for each point pair
        angles: Array of shape (n_samples, 2) with angles in [0, π)
        labels: Optional array of shape (n_samples,) with sample labels (1=PV, 0=symmetric)
        title: Plot title
        line_length: Length of orientation line segments
        figsize: Figure size tuple
        cmap: Colormap for z-axis coloring
        save_path: If provided, save figure to this path
        show_colorbar: Whether to show z-axis colorbar
        subset_size: If provided, randomly select this many samples for clearer visualization
        seed: Random seed for subset selection
        
    Returns:
        fig, ax: The matplotlib figure and axis objects
    """
    # Flatten to get all points
    n_samples = positions.shape[0]
    
    # Optionally select a subset for clearer visualization
    if subset_size is not None and subset_size < n_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_samples, size=subset_size, replace=False)
        positions = positions[indices]
        angles = angles[indices]
        if labels is not None:
            labels = labels[indices]
        n_samples = subset_size
    
    # Get x, y, z for all points
    x = positions[:, :, 0].flatten()  # (n_samples * 2,)
    y = positions[:, :, 1].flatten()  # (n_samples * 2,)
    z = positions[:, :, 2].flatten()  # (n_samples * 2,)
    phi = angles.flatten()  # (n_samples * 2,)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize z for coloring
    z_min, z_max = z.min(), z.max()
    norm = Normalize(vmin=z_min, vmax=z_max)
    colormap = cm.get_cmap(cmap)
    
    # Create line segments for spin-2 orientations
    # For spin-2, the orientation is a headless line, so we draw from -d to +d
    half_len = line_length / 2
    
    segments = []
    colors = []
    
    for i in range(len(x)):
        # Direction vector for the spin-2 orientation
        dx = half_len * np.cos(phi[i])
        dy = half_len * np.sin(phi[i])
        
        # Line segment endpoints (symmetric around point)
        x0, y0 = x[i] - dx, y[i] - dy
        x1, y1 = x[i] + dx, y[i] + dy
        
        segments.append([(x0, y0), (x1, y1)])
        colors.append(colormap(norm(z[i])))
    
    # Create line collection
    lc = LineCollection(segments, colors=colors, linewidths=1.5, alpha=0.8)
    ax.add_collection(lc)
    
    # Also scatter the points themselves
    scatter = ax.scatter(x, y, c=z, cmap=cmap, s=15, alpha=0.6, edgecolors='none')
    
    # Add colorbar for z-axis
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, label='z (line-of-sight)')
        cbar.ax.tick_params(labelsize=10)
    
    # Draw lines connecting paired points
    for i in range(n_samples):
        p1 = positions[i, 0, :2]
        p2 = positions[i, 1, :2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.1, linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.autoscale_view()
    
    # Add grid
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
    """
    Visualize the parity-violating dataset.
    
    Shows only the "real" PV samples (label=1) to highlight the parity-violating structure.
    
    Args:
        n_samples: Total number of samples to generate
        alpha: Parity violation angle offset
        seed: Random seed
        subset_size: Number of samples to display for clearer visualization
        save_path: Path to save the figure
        **kwargs: Additional arguments passed to plot_spin2_orientations
        
    Returns:
        fig, ax: The matplotlib figure and axis objects
    """
    # Generate dataset
    dataset = ParityViolationDataset(n_samples=n_samples, alpha=alpha, seed=seed)
    
    # The dataset generates real PV samples (label=1) first, then symmetrized (label=0).
    # Select only real PV samples from the first half.
    n_pv = n_samples // 2
    positions = dataset.positions_np[:n_pv]
    angles = dataset.angles_np[:n_pv]
    labels = dataset.labels[:n_pv].numpy()
    
    # Verify we have the correct samples (all should be label=1)
    assert (labels == 1).all(), "Expected first half to be real PV samples (label=1)"
    
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
    """
    Visualize the parity-symmetric (null test) dataset.
    
    Shows random angle samples with no parity violation for comparison.
    
    Args:
        n_samples: Total number of samples to generate
        seed: Random seed
        subset_size: Number of samples to display for clearer visualization
        save_path: Path to save the figure
        **kwargs: Additional arguments passed to plot_spin2_orientations
        
    Returns:
        fig, ax: The matplotlib figure and axis objects
    """
    # Generate dataset
    dataset = ParitySymmetricDataset(n_samples=n_samples, seed=seed)
    
    # For the symmetric dataset, all samples have random angles (no PV structure).
    # Use first half of samples for visualization consistency.
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
    """
    Visualize the parity-violating structure by showing pairs color-coded by delta_z sign.
    
    This visualization helps see how the angle difference correlates with z-ordering,
    which is the key signature of parity violation.
    
    Args:
        n_samples: Number of samples to generate
        alpha: Parity violation angle offset
        seed: Random seed
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        fig, axes: The matplotlib figure and axis objects
    """
    dataset = ParityViolationDataset(n_samples=n_samples, alpha=alpha, seed=seed)
    
    # Get only PV samples (first half)
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
    
    # Plot 2: Histogram of sin(2Δφ) for positive vs negative delta_z
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
    
    # Plot 3: Scatter showing pairs in 2D with color by delta_z
    ax3 = axes[2]
    n_show = min(30, n_pv)
    for i in range(n_show):
        p1 = positions[i, 0]
        p2 = positions[i, 1]
        color = 'red' if delta_z[i] > 0 else 'blue'
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, alpha=0.5, linewidth=1)
        
        # Draw orientation arrows
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


def plot_training_convergence(
    train_losses: list,
    val_losses: list,
    val_accuracies: list = None,
    title: str = "Training Convergence",
    save_path: str = None,
    figsize: tuple = (12, 4)
):
    """
    Plot training loss convergence over epochs.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: Optional list of validation accuracies per epoch
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        fig, axes: The matplotlib figure and axis objects
    """
    epochs = range(1, len(train_losses) + 1)
    
    n_plots = 2 if val_accuracies is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot losses
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (BCE)', fontsize=12)
    ax1.set_title('Loss vs Epoch', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy if provided
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
    """
    Compare training convergence between PV and control experiments.
    
    Args:
        pv_results: Results dictionary from run_experiment (with parity violation)
        control_results: Results dictionary from run_control_experiment (null test)
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        fig, axes: The matplotlib figure and axis objects
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(pv_results['train_losses']) + 1)
    
    # Plot 1: Training losses comparison
    ax1 = axes[0]
    ax1.plot(epochs, pv_results['train_losses'], 'b-', label='PV Train', linewidth=2)
    ax1.plot(epochs, control_results['train_losses'], 'r--', label='Control Train', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation losses comparison
    ax2 = axes[1]
    ax2.plot(epochs, pv_results['val_losses'], 'b-', label='PV Val', linewidth=2)
    ax2.plot(epochs, control_results['val_losses'], 'r--', label='Control Val', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation accuracy comparison
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


if __name__ == '__main__':
    import os
    
    # Create output directory
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations for spin-2 parity violation experiment...")
    
    # 1. Visualize PV dataset
    print("\n1. Visualizing parity-violating dataset...")
    fig1, ax1 = plot_pv_dataset(
        n_samples=500,
        alpha=0.3,
        subset_size=60,
        save_path=os.path.join(output_dir, 'pv_dataset.png')
    )
    
    # 2. Visualize null test dataset
    print("\n2. Visualizing null test dataset...")
    fig2, ax2 = plot_null_dataset(
        n_samples=500,
        subset_size=60,
        save_path=os.path.join(output_dir, 'null_dataset.png')
    )
    
    # 3. Visualize PV structure in detail
    print("\n3. Visualizing parity-violating structure...")
    fig3, ax3 = plot_pv_structure(
        n_samples=400,
        alpha=0.3,
        save_path=os.path.join(output_dir, 'pv_structure.png')
    )
    
    # 4. Run experiments and visualize convergence
    print("\n4. Running training experiments...")
    from train import run_experiment, run_control_experiment
    
    # Run smaller experiments for visualization
    pv_results = run_experiment(
        n_train=2000, n_val=500, n_test=500,
        alpha=0.3, n_epochs=30, verbose=True
    )
    
    control_results = run_control_experiment(
        n_train=2000, n_val=500, n_test=500,
        n_epochs=30, verbose=True
    )
    
    # 5. Plot training convergence for PV experiment
    print("\n5. Plotting training convergence...")
    fig4, ax4 = plot_training_convergence(
        train_losses=pv_results['train_losses'],
        val_losses=pv_results['val_losses'],
        val_accuracies=pv_results['val_accuracies'],
        title="PV Detection Training Convergence (Spin-2)",
        save_path=os.path.join(output_dir, 'pv_convergence.png')
    )
    
    # 6. Plot comparison between PV and control
    print("\n6. Plotting PV vs Control comparison...")
    fig5, ax5 = plot_comparison(
        pv_results=pv_results,
        control_results=control_results,
        save_path=os.path.join(output_dir, 'pv_vs_control.png')
    )
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory")
    print(f"\nFinal Results:")
    print(f"  PV Test Accuracy: {pv_results['test_accuracy']:.4f}")
    print(f"  Control Test Accuracy: {control_results['test_accuracy']:.4f}")
    
    plt.show()

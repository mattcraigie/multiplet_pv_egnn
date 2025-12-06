"""
Threshold Search for finding detection boundaries.

This module implements boundary search to find detection thresholds:
- Finds the minimum f_pv required for detection at each dataset size
- Uses iterative binary search to localize the detection boundary
- Answers: "What parity strength can my model detect with X samples?"

Default dataset sizes: 10^2, 10^3, 10^4, 10^5 (100, 1000, 10000, 100000)

Results include:
- Detection boundary curve
- Minimum f_pv for detection at each dataset size
- Confidence heatmap showing detection confidence across all tested points

Usage:
    python -m experiments.threshold_search
    python -m experiments.threshold_search --config configs/threshold_search_config.yaml
    python -m experiments.threshold_search --plot-only boundary_results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Import from experiments package - works when running as a module (-m experiments.threshold_search)
try:
    from experiments.basic_train import (
        run_bootstrap_statistical_test,
        DEFAULT_MODEL_TYPE,
        MODEL_TYPE_FRAME_ALIGNED,
    )
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments.basic_train import (
        run_bootstrap_statistical_test,
        DEFAULT_MODEL_TYPE,
        MODEL_TYPE_FRAME_ALIGNED,
    )


# Default configuration
DEFAULT_CONFIG = {
    # Boundary search parameters
    'num_train_val': [100, 1000, 10000, 100000],
    'f_pv_min': 0.01,
    'f_pv_max': 0.2,
    'boundary_max_depth': 8,
    'boundary_n_splits': 4,
    'train_val_split': 0.8,
    
    # Fixed test set size
    'n_test': 100000,
    
    # Model parameters - Default to Frame-Aligned model
    'model_type': DEFAULT_MODEL_TYPE,
    'alpha': 0.3,
    'hidden_dim': 64,
    'n_layers': 2,
    'num_slots': 32,
    'num_hops': 3,
    
    # Training parameters
    'batch_size': 64,
    'n_epochs': 100,
    'lr': 1e-3,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 1e-4,
    
    # Statistical test parameters
    'n_bootstrap': 1000,
    'confidence_level': 0.95,
    'seed': 42,
    
    # Output
    'output_dir': 'threshold_search_results',
    'verbose': True
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from a YAML file or return defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config.update(user_config)
    
    return config


def save_config(config: dict, output_path: str):
    """Save configuration to a YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def detection_function(
    n_train_val: int,
    f_pv: float,
    config: dict
) -> tuple:
    """
    Test whether parity violation is detected for given parameters.
    
    This function trains a model and returns detection status and confidence.
    
    Args:
        n_train_val: Number of training + validation samples
        f_pv: Parity violation fraction
        config: Configuration dictionary with training parameters
        
    Returns:
        Tuple of (detected: int, detection_confidence: float, test_accuracy: float)
        - detected: 1 if parity violation detected, 0 otherwise
        - detection_confidence: confidence level (0.0 to 1.0)
        - test_accuracy: test accuracy
    """
    train_split = config['train_val_split']
    n_train = int(n_train_val * train_split)
    n_val = n_train_val - n_train
    
    try:
        run_results = run_bootstrap_statistical_test(
            n_train=n_train,
            n_val=n_val,
            n_test=config['n_test'],
            alpha=config['alpha'],
            f_pv=f_pv,
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            lr=config['lr'],
            seed=config['seed'],
            n_bootstrap=config['n_bootstrap'],
            confidence_level=config['confidence_level'],
            verbose=config.get('verbose', False),
            early_stopping_patience=config['early_stopping_patience'],
            early_stopping_min_delta=config['early_stopping_min_delta'],
            model_type=config.get('model_type', DEFAULT_MODEL_TYPE),
            num_slots=config.get('num_slots', 32),
            num_hops=config.get('num_hops', 3)
        )
        detected = 1 if run_results['parity_violation_detected'] else 0
        detection_confidence = run_results['detection_confidence']
        test_accuracy = run_results['test_accuracy']
        return detected, detection_confidence, test_accuracy
    except Exception as e:
        print(f"Error at n_train_val={n_train_val}, f_pv={f_pv}: {e}")
        return 0, 0.0, 0.5


def find_boundary_curve(
    num_train_val_values: list,
    f_pv_range: tuple,
    config: dict,
    max_depth: int = 8,
    n_splits: int = 4,
) -> tuple:
    """
    Find the detection boundary curve using binary search.
    
    For each dataset size (x = num_train_val), finds the minimum f_pv (y)
    at which parity violation can be detected. Uses iterative refinement
    to localize the transition between detection and no detection.
    
    This answers the question: "What parity strength can my model detect
    with an X sized dataset of points?"
    
    Args:
        num_train_val_values: List of dataset sizes to test
        f_pv_range: Tuple (f_pv_min, f_pv_max) defining the search range
        config: Configuration dictionary with training parameters
        max_depth: Maximum depth of binary search refinement
        n_splits: Number of splits per refinement level
        
    Returns:
        Tuple of (num_train_val_values, f_pv_boundary, tested_points) where
        - f_pv_boundary[i] is the estimated boundary f_pv for num_train_val_values[i]
        - tested_points is a list of dicts with (n_train_val, f_pv, detected, confidence, accuracy)
    """
    f_pv_min, f_pv_max = f_pv_range
    
    xs = np.array(num_train_val_values)
    ys_boundary = np.full(len(xs), np.nan, dtype=float)
    
    # Track all tested points for confidence heatmap
    tested_points = []
    
    verbose = config.get('verbose', True)
    
    for idx, n_train_val in enumerate(xs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Finding boundary for n_train_val={n_train_val}")
            print(f"{'='*60}")
        
        lo, hi = f_pv_min, f_pv_max
        found_interval = None
        
        for depth in range(max_depth):
            f_pv_samples = np.linspace(lo, hi, n_splits + 1)
            
            if verbose:
                print(f"  Depth {depth}: searching f_pv in [{lo:.4f}, {hi:.4f}]")
            
            detection_vals = []
            for f_pv in f_pv_samples:
                detected, confidence, accuracy = detection_function(n_train_val, f_pv, config)
                detection_vals.append(detected)
                
                # Store the tested point
                tested_points.append({
                    'n_train_val': int(n_train_val),
                    'f_pv': float(f_pv),
                    'detected': int(detected),
                    'confidence': float(confidence),
                    'accuracy': float(accuracy)
                })
                
                if verbose:
                    status = "DETECTED" if detected else "not detected"
                    print(f"    f_pv={f_pv:.4f}: {status} (confidence={confidence*100:.1f}%)")
            
            transition_index = None
            for i in range(n_splits):
                if detection_vals[i] != detection_vals[i + 1]:
                    transition_index = i
                    break
            
            if transition_index is None:
                found_interval = None
                break
            
            lo = f_pv_samples[transition_index]
            hi = f_pv_samples[transition_index + 1]
            found_interval = (lo, hi)
        
        if found_interval is not None:
            lo, hi = found_interval
            ys_boundary[idx] = 0.5 * (lo + hi)
            if verbose:
                print(f"  Boundary found: f_pv ≈ {ys_boundary[idx]:.4f}")
        else:
            if verbose:
                print(f"  No boundary found in range [{f_pv_min}, {f_pv_max}]")
    
    return xs, ys_boundary, tested_points


def run_threshold_search(config: dict) -> dict:
    """
    Run threshold search to find detection boundaries for each dataset size.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with threshold search results
    """
    num_train_val_list = config.get('num_train_val', DEFAULT_CONFIG['num_train_val'])
    f_pv_range = (config.get('f_pv_min', 0.01), config.get('f_pv_max', 0.2))
    max_depth = config.get('boundary_max_depth', 8)
    n_splits = config.get('boundary_n_splits', 4)
    verbose = config.get('verbose', True)
    
    if verbose:
        print("\n" + "="*60)
        print("Threshold Search: Finding Detection Boundaries")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  num_train_val: {num_train_val_list}")
        print(f"  f_pv_range: {f_pv_range}")
        print(f"  max_depth: {max_depth}")
        print(f"  n_splits: {n_splits}")
        print(f"  model_type: {config.get('model_type', DEFAULT_MODEL_TYPE)}")
    
    xs, ys_boundary, tested_points = find_boundary_curve(
        num_train_val_values=num_train_val_list,
        f_pv_range=f_pv_range,
        config=config,
        max_depth=max_depth,
        n_splits=n_splits
    )
    
    results = {
        'config': config,
        'num_train_val': xs.tolist(),
        'f_pv_boundary': ys_boundary.tolist(),
        'f_pv_range': f_pv_range,
        'max_depth': max_depth,
        'n_splits': n_splits,
        'tested_points': tested_points,  # All tested points with confidence data
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def save_results(results: dict, output_dir: str):
    """Save threshold search results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'boundary_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def load_results(results_path: str) -> dict:
    """Load threshold search results from a JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def plot_boundary_curve(
    results: dict,
    output_dir: str,
    figsize: tuple = (10, 8)
):
    """
    Plot the detection boundary curve.
    
    Shows the minimum f_pv required for detection as a function of dataset size.
    
    Args:
        results: Threshold search results dictionary
        output_dir: Output directory for saving the plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_train_val = np.array(results['num_train_val'])
    f_pv_boundary = np.array(results['f_pv_boundary'])
    
    # Filter out NaN values
    valid_mask = ~np.isnan(f_pv_boundary)
    x_valid = num_train_val[valid_mask]
    y_valid = f_pv_boundary[valid_mask]
    
    # Plot boundary curve
    if len(x_valid) > 0:
        ax.plot(x_valid, y_valid, 'b-o', linewidth=2, markersize=8, label='Detection Boundary')
        
        # Fill regions
        ax.fill_between(x_valid, y_valid, 1.0, alpha=0.3, color='green', label='Detected Region')
        ax.fill_between(x_valid, 0, y_valid, alpha=0.3, color='red', label='Not Detected Region')
    
    # Mark points where boundary was not found
    invalid_mask = np.isnan(f_pv_boundary)
    if np.any(invalid_mask):
        ax.scatter(num_train_val[invalid_mask], 
                  np.full(np.sum(invalid_mask), 0.5),
                  marker='x', s=100, c='gray', label='Boundary not found')
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    ax.set_xlabel('Number of Train+Val Samples', fontsize=12)
    ax.set_ylabel('f_pv (Parity Violation Fraction)', fontsize=12)
    ax.set_title('Detection Boundary: Minimum f_pv for Detection\n'
                 '(What parity strength can my model detect with X samples?)', fontsize=14)
    
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'boundary_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved boundary curve to {output_path}")
    
    plt.close()


def plot_confidence_heatmap(
    results: dict,
    output_dir: str,
    figsize: tuple = (12, 8)
):
    """
    Plot a confidence heatmap for the threshold search results.
    
    This creates an interpolated heatmap showing detection confidence
    across the non-uniform grid of tested points.
    
    Args:
        results: Threshold search results dictionary with 'tested_points'
        output_dir: Output directory for saving the plot
        figsize: Figure size
    """
    from scipy.interpolate import griddata
    
    tested_points = results.get('tested_points', [])
    if not tested_points:
        print("No tested points available for confidence heatmap")
        return
    
    # Extract data from tested points
    n_train_vals = np.array([p['n_train_val'] for p in tested_points])
    f_pvs = np.array([p['f_pv'] for p in tested_points])
    confidences = np.array([p['confidence'] for p in tested_points])
    
    # Use log scale for n_train_val
    log_n = np.log10(n_train_vals)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a regular grid for interpolation
    n_grid = 100
    log_n_min, log_n_max = log_n.min(), log_n.max()
    f_pv_min, f_pv_max = f_pvs.min(), f_pvs.max()
    
    log_n_grid = np.linspace(log_n_min, log_n_max, n_grid)
    f_pv_grid = np.linspace(f_pv_min, f_pv_max, n_grid)
    log_n_mesh, f_pv_mesh = np.meshgrid(log_n_grid, f_pv_grid)
    
    # Interpolate confidence values onto the grid
    confidence_grid = griddata(
        (log_n, f_pvs),
        confidences,
        (log_n_mesh, f_pv_mesh),
        method='linear',
        fill_value=np.nan
    )
    
    # Also try with 'nearest' for areas outside the convex hull
    confidence_grid_nearest = griddata(
        (log_n, f_pvs),
        confidences,
        (log_n_mesh, f_pv_mesh),
        method='nearest'
    )
    
    # Use nearest where linear failed
    mask = np.isnan(confidence_grid)
    confidence_grid[mask] = confidence_grid_nearest[mask]
    
    # Plot heatmap
    im = ax.imshow(
        confidence_grid * 100,  # Convert to percentage
        extent=[log_n_min, log_n_max, f_pv_min, f_pv_max],
        origin='lower',
        aspect='auto',
        cmap='RdYlGn',
        vmin=0,
        vmax=100
    )
    
    # Plot actual data points
    ax.scatter(log_n, f_pvs, c='black', s=20, alpha=0.6, marker='o', label='Tested Points')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Detection Confidence (%)', fontsize=12)
    
    # Set x-axis ticks to show actual n_train_val values
    unique_log_n = np.unique(log_n)
    ax.set_xticks(unique_log_n)
    ax.set_xticklabels([f'$10^{{{int(v)}}}$' for v in unique_log_n], fontsize=10)
    
    ax.set_xlabel('Number of Train+Val Samples', fontsize=12)
    ax.set_ylabel('f_pv (Parity Violation Fraction)', fontsize=12)
    ax.set_title('Detection Confidence Heatmap\n(Interpolated from Binary Search Results)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'confidence_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved confidence heatmap to {output_path}")
    
    plt.close()


def plot_accuracy_scatter(
    results: dict,
    output_dir: str,
    figsize: tuple = (12, 8)
):
    """
    Plot a scatter plot of test accuracy for all tested points.
    
    Args:
        results: Threshold search results dictionary with 'tested_points'
        output_dir: Output directory for saving the plot
        figsize: Figure size
    """
    tested_points = results.get('tested_points', [])
    if not tested_points:
        print("No tested points available for accuracy scatter plot")
        return
    
    # Extract data from tested points
    n_train_vals = np.array([p['n_train_val'] for p in tested_points])
    f_pvs = np.array([p['f_pv'] for p in tested_points])
    accuracies = np.array([p['accuracy'] for p in tested_points])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with accuracy as color
    sc = ax.scatter(
        n_train_vals, f_pvs,
        c=accuracies * 100,  # Convert to percentage
        s=100,
        cmap='viridis',
        vmin=50,
        vmax=100,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Test Accuracy (%)', fontsize=12)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    ax.set_xlabel('Number of Train+Val Samples', fontsize=12)
    ax.set_ylabel('f_pv (Parity Violation Fraction)', fontsize=12)
    ax.set_title('Test Accuracy for All Tested Points', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'accuracy_scatter.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved accuracy scatter plot to {output_path}")
    
    plt.close()


def print_summary(results: dict):
    """Print summary of threshold search results."""
    print("\nThreshold Summary (minimum f_pv for detection):")
    for n, f in zip(results['num_train_val'], results['f_pv_boundary']):
        if f is None or (isinstance(f, float) and np.isnan(f)):
            print(f"  n_train_val={n:>7,}: boundary not found")
        else:
            print(f"  n_train_val={n:>7,}: f_pv >= {f:.4f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run threshold search to find detection boundaries',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (overrides config)')
    parser.add_argument('--plot-only', type=str, default=None,
                        help='Skip search and only plot from existing boundary_results.json')
    parser.add_argument('--generate-default-config', type=str, default=None,
                        help='Generate default config file at the specified path and exit')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Generate default config if requested
    if args.generate_default_config:
        save_config(DEFAULT_CONFIG, args.generate_default_config)
        print(f"Default configuration saved to {args.generate_default_config}")
        exit(0)
    
    # Plot-only mode
    if args.plot_only:
        print(f"Loading results from {args.plot_only}")
        results = load_results(args.plot_only)
        output_dir = os.path.dirname(args.plot_only)
        plot_boundary_curve(results, output_dir)
        if 'tested_points' in results:
            plot_confidence_heatmap(results, output_dir)
            plot_accuracy_scatter(results, output_dir)
        print_summary(results)
        exit(0)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.quiet:
        config['verbose'] = False
    
    output_dir = config['output_dir']
    
    # Run threshold search
    print("="*60)
    print("Threshold Search: Finding Detection Boundaries")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  num_train_val: {config['num_train_val']}")
    print(f"  f_pv_range: ({config.get('f_pv_min', 0.01)}, {config.get('f_pv_max', 0.2)})")
    print(f"  train_val_split: {config['train_val_split']}")
    print(f"  n_test: {config['n_test']}")
    print(f"  model_type: {config.get('model_type', DEFAULT_MODEL_TYPE)}")
    print(f"  output_dir: {output_dir}")
    
    # Save config
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, os.path.join(output_dir, 'config.yaml'))
    
    # Run threshold search
    print("\nStarting threshold search...")
    results = run_threshold_search(config)
    
    # Save results
    save_results(results, output_dir)
    
    # Generate plot
    print("\nGenerating plots...")
    plot_boundary_curve(results, output_dir)
    plot_confidence_heatmap(results, output_dir)
    plot_accuracy_scatter(results, output_dir)
    
    print("\n" + "="*60)
    print("Threshold Search Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - config.yaml")
    print(f"  - boundary_results.json")
    print(f"  - boundary_curve.png")
    print(f"  - confidence_heatmap.png")
    print(f"  - accuracy_scatter.png")
    
    # Print summary
    print_summary(results)

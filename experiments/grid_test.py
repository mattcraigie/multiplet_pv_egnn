"""
Grid Test for comparing data amount to detection capability.

This module implements a grid test over:
1. Number of train/val samples
2. Parity violation fraction (f_pv)

The test evaluates detection significance across parameter combinations
to understand how data quantity affects detection capability.

Default dataset sizes: 10^2, 10^3, 10^4, 10^5 (100, 1000, 10000, 100000)

Results include:
- Detection significance heatmaps
- Binary detection threshold heatmaps
- Test accuracy heatmaps

Usage:
    python -m experiments.grid_test
    python -m experiments.grid_test --config grid_test_config.yaml
    python -m experiments.grid_test --plot-only results.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.basic_train import (
    run_bootstrap_statistical_test,
    DEFAULT_MODEL_TYPE,
    MODEL_TYPE_FRAME_ALIGNED,
    MODEL_TYPE_EGNN,
)


# Heatmap color thresholds for text visibility
HEATMAP_TEXT_DARK_THRESHOLD_LOW = 50
HEATMAP_TEXT_DARK_THRESHOLD_HIGH = 80
ACCURACY_HEATMAP_TEXT_THRESHOLD = 70


# Default configuration
DEFAULT_CONFIG = {
    # Grid parameters
    'num_train_val': [100, 1000, 10000, 100000],
    'f_pv_values': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    'train_val_split': 0.8,
    
    # Fixed test set size
    'n_test': 100000,
    
    # Model parameters - Default to Frame-Aligned model
    'model_type': DEFAULT_MODEL_TYPE,
    'alpha': 0.3,
    'hidden_dim': 16,
    'n_layers': 2,
    'num_slots': 8,
    'num_hops': 2,
    
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
    'output_dir': 'grid_test_results',
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


def run_grid_test(config: dict) -> dict:
    """
    Run grid test over num_train_val and f_pv values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with results for each grid point
    """
    num_train_val_list = config['num_train_val']
    f_pv_list = config['f_pv_values']
    train_split = config['train_val_split']
    n_test = config['n_test']
    verbose = config['verbose']
    
    results = {
        'config': config,
        'grid_results': [],
        'detection_confidence_matrix': np.zeros((len(num_train_val_list), len(f_pv_list))),
        'detection_binary_matrix': np.zeros((len(num_train_val_list), len(f_pv_list))),
        'test_accuracy_matrix': np.zeros((len(num_train_val_list), len(f_pv_list))),
        'num_train_val': num_train_val_list,
        'f_pv_values': f_pv_list
    }
    
    total_runs = len(num_train_val_list) * len(f_pv_list)
    run_count = 0
    
    for i, n_train_val in enumerate(num_train_val_list):
        n_train = int(n_train_val * train_split)
        n_val = n_train_val - n_train
        
        for j, f_pv in enumerate(f_pv_list):
            run_count += 1
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Grid Test Progress: {run_count}/{total_runs}")
                print(f"n_train_val={n_train_val}, f_pv={f_pv}")
                print(f"n_train={n_train}, n_val={n_val}, n_test={n_test}")
                print(f"{'='*60}")
            
            try:
                run_results = run_bootstrap_statistical_test(
                    n_train=n_train,
                    n_val=n_val,
                    n_test=n_test,
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
                    verbose=verbose,
                    early_stopping_patience=config['early_stopping_patience'],
                    early_stopping_min_delta=config['early_stopping_min_delta'],
                    model_type=config.get('model_type', DEFAULT_MODEL_TYPE),
                    num_slots=config.get('num_slots', 8),
                    num_hops=config.get('num_hops', 2)
                )
                
                grid_result = {
                    'n_train_val': int(n_train_val),
                    'n_train': int(n_train),
                    'n_val': int(n_val),
                    'f_pv': float(f_pv),
                    'test_accuracy': float(run_results['test_accuracy']),
                    'ci_lower': float(run_results['ci_lower']),
                    'ci_upper': float(run_results['ci_upper']),
                    'p_value': float(run_results['p_value']),
                    'parity_violation_detected': bool(run_results['parity_violation_detected']),
                    'detection_confidence': float(run_results['detection_confidence']),
                    'epochs_trained': int(run_results.get('epochs_trained', config['n_epochs']))
                }
                
                results['grid_results'].append(grid_result)
                results['detection_confidence_matrix'][i, j] = run_results['detection_confidence']
                results['detection_binary_matrix'][i, j] = 1.0 if run_results['parity_violation_detected'] else 0.0
                results['test_accuracy_matrix'][i, j] = run_results['test_accuracy']
                
                if verbose:
                    print(f"\nResult: accuracy={run_results['test_accuracy']:.4f}, "
                          f"detection_confidence={run_results['detection_confidence']*100:.1f}%")
                    
            except Exception as e:
                print(f"Error at n_train_val={n_train_val}, f_pv={f_pv}: {e}")
                results['grid_results'].append({
                    'n_train_val': int(n_train_val),
                    'f_pv': float(f_pv),
                    'error': str(e)
                })
                results['detection_confidence_matrix'][i, j] = np.nan
                results['detection_binary_matrix'][i, j] = np.nan
                results['test_accuracy_matrix'][i, j] = np.nan
    
    return results


def save_results(results: dict, output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    json_results = {
        'config': results['config'],
        'grid_results': results['grid_results'],
        'num_train_val': results['num_train_val'],
        'f_pv_values': results['f_pv_values'],
        'detection_confidence_matrix': results['detection_confidence_matrix'].tolist(),
        'detection_binary_matrix': results['detection_binary_matrix'].tolist(),
        'test_accuracy_matrix': results['test_accuracy_matrix'].tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = os.path.join(output_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def load_results(results_path: str) -> dict:
    """Load results from a JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    results['detection_confidence_matrix'] = np.array(results['detection_confidence_matrix'])
    results['detection_binary_matrix'] = np.array(results['detection_binary_matrix'])
    results['test_accuracy_matrix'] = np.array(results['test_accuracy_matrix'])
    
    return results


def plot_detection_significance_heatmap(
    results: dict,
    output_dir: str,
    figsize: tuple = (10, 8)
):
    """Plot detection significance (percentage) heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    matrix = results['detection_confidence_matrix'] * 100
    num_train_val = results['num_train_val']
    f_pv_values = results['f_pv_values']
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Detection Confidence (%)', fontsize=12)
    
    ax.set_xticks(range(len(f_pv_values)))
    ax.set_xticklabels([f'{v:.2f}' for v in f_pv_values], fontsize=10)
    ax.set_yticks(range(len(num_train_val)))
    ax.set_yticklabels([f'{v:,}' for v in num_train_val], fontsize=10)
    
    ax.set_xlabel('f_pv (Parity Violation Fraction)', fontsize=12)
    ax.set_ylabel('Number of Train+Val Samples', fontsize=12)
    ax.set_title('Detection Significance Heatmap\n(% Confidence of Parity Violation Detection)', fontsize=14)
    
    for i in range(len(num_train_val)):
        for j in range(len(f_pv_values)):
            value = matrix[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < HEATMAP_TEXT_DARK_THRESHOLD_LOW or value > HEATMAP_TEXT_DARK_THRESHOLD_HIGH else 'black'
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'detection_significance_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved detection significance heatmap to {output_path}")
    
    plt.close()


def plot_detection_binary_heatmap(
    results: dict,
    output_dir: str,
    figsize: tuple = (10, 8)
):
    """Plot binary detection threshold heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    matrix = results['detection_binary_matrix']
    num_train_val = results['num_train_val']
    f_pv_values = results['f_pv_values']
    
    colors = ['#d73027', '#1a9850']
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['Not Detected', 'Detected'])
    cbar.set_label('Detection Status', fontsize=12)
    
    ax.set_xticks(range(len(f_pv_values)))
    ax.set_xticklabels([f'{v:.2f}' for v in f_pv_values], fontsize=10)
    ax.set_yticks(range(len(num_train_val)))
    ax.set_yticklabels([f'{v:,}' for v in num_train_val], fontsize=10)
    
    ax.set_xlabel('f_pv (Parity Violation Fraction)', fontsize=12)
    ax.set_ylabel('Number of Train+Val Samples', fontsize=12)
    
    confidence_level = results['config'].get('confidence_level', 0.95) * 100
    ax.set_title(f'Binary Detection Heatmap\n(Parity Violation Detected at {confidence_level:.0f}% Confidence Level)', 
                fontsize=14)
    
    for i in range(len(num_train_val)):
        for j in range(len(f_pv_values)):
            value = matrix[i, j]
            if not np.isnan(value):
                text = '✓' if value == 1 else '✗'
                ax.text(j, i, text, ha='center', va='center',
                       color='white', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'detection_binary_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved binary detection heatmap to {output_path}")
    
    plt.close()


def plot_accuracy_heatmap(
    results: dict,
    output_dir: str,
    figsize: tuple = (10, 8)
):
    """Plot test accuracy heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    matrix = results['test_accuracy_matrix'] * 100
    num_train_val = results['num_train_val']
    f_pv_values = results['f_pv_values']
    
    im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=50, vmax=100)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Test Accuracy (%)', fontsize=12)
    
    ax.set_xticks(range(len(f_pv_values)))
    ax.set_xticklabels([f'{v:.2f}' for v in f_pv_values], fontsize=10)
    ax.set_yticks(range(len(num_train_val)))
    ax.set_yticklabels([f'{v:,}' for v in num_train_val], fontsize=10)
    
    ax.set_xlabel('f_pv (Parity Violation Fraction)', fontsize=12)
    ax.set_ylabel('Number of Train+Val Samples', fontsize=12)
    ax.set_title('Test Accuracy Heatmap', fontsize=14)
    
    for i in range(len(num_train_val)):
        for j in range(len(f_pv_values)):
            value = matrix[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < ACCURACY_HEATMAP_TEXT_THRESHOLD else 'black'
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test_accuracy_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved test accuracy heatmap to {output_path}")
    
    plt.close()


def generate_plots(results: dict, output_dir: str):
    """Generate all heatmap plots from results."""
    plot_detection_significance_heatmap(results, output_dir)
    plot_detection_binary_heatmap(results, output_dir)
    plot_accuracy_heatmap(results, output_dir)
    print(f"\nAll plots saved to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run grid test for parity violation detection capability',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (overrides config)')
    parser.add_argument('--plot-only', type=str, default=None,
                        help='Skip grid test and only generate plots from existing results.json')
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
        generate_plots(results, output_dir)
        exit(0)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    if args.quiet:
        config['verbose'] = False
    
    output_dir = config['output_dir']
    
    # Run grid test
    print("="*60)
    print("Grid Test: Data Amount vs Detection Capability")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  num_train_val: {config['num_train_val']}")
    print(f"  f_pv_values: {config['f_pv_values']}")
    print(f"  train_val_split: {config['train_val_split']}")
    print(f"  n_test: {config['n_test']}")
    print(f"  model_type: {config.get('model_type', DEFAULT_MODEL_TYPE)}")
    print(f"  output_dir: {output_dir}")
    
    # Save config
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, os.path.join(output_dir, 'config.yaml'))
    
    # Run grid test
    print("\nStarting grid test...")
    results = run_grid_test(config)
    
    # Save results
    save_results(results, output_dir)
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results, output_dir)
    
    print("\n" + "="*60)
    print("Grid Test Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - config.yaml")
    print(f"  - results.json")
    print(f"  - detection_significance_heatmap.png")
    print(f"  - detection_binary_heatmap.png")
    print(f"  - test_accuracy_heatmap.png")

"""
Model Comparison Experiment for 3D Parity Violation Detection.

This experiment compares two model types:
1. EGNN (original) - EGNN-like message passing classifier
2. Frame-Aligned GNN - Frame-aligned message passing with latent slots

The comparison is done with configurable parameters:
- Number of training+validation points
- f_pv (parity violation fraction)
- Number of random seeds for averaging

Usage:
    python -m experiments.compare_models
    python -m experiments.compare_models --n-points 10000 --f-pv 0.05 --n-seeds 3
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import from experiments package - works when running as a module (-m experiments.compare_models)
try:
    from experiments.basic_train import (
        run_experiment,
        run_bootstrap_statistical_test,
        MODEL_TYPE_EGNN,
        MODEL_TYPE_FRAME_ALIGNED,
        SEED_MULTIPLIER,
    )
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments.basic_train import (
        run_experiment,
        run_bootstrap_statistical_test,
        MODEL_TYPE_EGNN,
        MODEL_TYPE_FRAME_ALIGNED,
        SEED_MULTIPLIER,
    )


def compare_models(
    n_points: int = 10000,
    f_pv: float = 0.05,
    n_seeds: int = 3,
    n_epochs: int = 50,
    hidden_dim: int = 32,
    n_layers: int = 2,
    num_slots: int = 8,
    num_hops: int = 2,
    alpha: float = 0.3,
    batch_size: int = 64,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    verbose: bool = True,
    output_dir: str = 'comparison_results'
):
    """
    Compare EGNN and Frame-Aligned models on parity violation detection.
    
    This experiment uses BOTH model types to compare their performance,
    unlike other experiments which default to Frame-Aligned only.
    
    Args:
        n_points: Total number of training + validation points
        f_pv: Fraction of parity-violating pairs
        n_seeds: Number of random seeds to average over
        n_epochs: Maximum number of training epochs
        hidden_dim: Hidden dimension for both models
        n_layers: Number of message passing layers (for EGNN)
        num_slots: Number of latent slots (for Frame-Aligned)
        num_hops: Number of hops (for Frame-Aligned)
        alpha: Parity violation angle parameter
        batch_size: Batch size for training
        lr: Learning rate
        early_stopping_patience: Early stopping patience
        verbose: Whether to print progress
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate train/val split (80/20)
    n_train = int(n_points * 0.8)
    n_val = n_points - n_train
    n_test = n_points
    
    print("="*70)
    print("Model Comparison: EGNN vs Frame-Aligned GNN")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Total points (train+val): {n_points}")
    print(f"  f_pv (parity violation fraction): {f_pv}")
    print(f"  Number of seeds: {n_seeds}")
    print(f"  alpha (PV angle): {alpha}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  batch_size: {batch_size}, lr: {lr}")
    print(f"  n_epochs: {n_epochs}, early_stopping_patience: {early_stopping_patience}")
    print()
    
    results = {
        'config': {
            'n_points': n_points,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'f_pv': f_pv,
            'n_seeds': n_seeds,
            'alpha': alpha,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'num_slots': num_slots,
            'num_hops': num_hops,
            'batch_size': batch_size,
            'lr': lr,
            'n_epochs': n_epochs,
            'early_stopping_patience': early_stopping_patience,
        },
        'egnn': {
            'accuracies': [],
            'val_losses': [],
            'epochs_trained': [],
        },
        'frame_aligned': {
            'accuracies': [],
            'val_losses': [],
            'epochs_trained': [],
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Run experiments for BOTH model types
    for model_type in [MODEL_TYPE_EGNN, MODEL_TYPE_FRAME_ALIGNED]:
        model_name = 'egnn' if model_type == MODEL_TYPE_EGNN else 'frame_aligned'
        print(f"\n{'='*70}")
        print(f"Testing Model: {model_type.upper()}")
        print(f"{'='*70}")
        
        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")
            
            exp_results = run_experiment(
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                alpha=alpha,
                f_pv=f_pv,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                batch_size=batch_size,
                n_epochs=n_epochs,
                lr=lr,
                seed=seed,
                verbose=verbose,
                early_stopping_patience=early_stopping_patience,
                model_type=model_type,
                num_slots=num_slots,
                num_hops=num_hops
            )
            
            results[model_name]['accuracies'].append(exp_results['test_accuracy'])
            results[model_name]['val_losses'].append(exp_results['val_losses'])
            results[model_name]['epochs_trained'].append(exp_results['epochs_trained'])
            
            print(f"  Test accuracy: {exp_results['test_accuracy']:.4f}")
            print(f"  Epochs trained: {exp_results['epochs_trained']}")
    
    # Compute summary statistics
    for model_name in ['egnn', 'frame_aligned']:
        accs = results[model_name]['accuracies']
        results[model_name]['mean_accuracy'] = float(np.mean(accs))
        results[model_name]['std_accuracy'] = float(np.std(accs))
        results[model_name]['mean_epochs'] = float(np.mean(results[model_name]['epochs_trained']))
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nConfiguration: {n_points} points, f_pv={f_pv}")
    print()
    
    for model_name, display_name in [('egnn', 'EGNN'), ('frame_aligned', 'Frame-Aligned GNN')]:
        m = results[model_name]
        print(f"{display_name}:")
        print(f"  Mean accuracy: {m['mean_accuracy']:.4f} ± {m['std_accuracy']:.4f}")
        print(f"  Individual runs: {[f'{a:.4f}' for a in m['accuracies']]}")
        print(f"  Mean epochs: {m['mean_epochs']:.1f}")
        print()
    
    # Compare
    egnn_mean = results['egnn']['mean_accuracy']
    fa_mean = results['frame_aligned']['mean_accuracy']
    diff = fa_mean - egnn_mean
    
    print(f"Difference (Frame-Aligned - EGNN): {diff:+.4f}")
    
    if diff > 0.01:
        print("→ Frame-Aligned GNN performs BETTER")
    elif diff < -0.01:
        print("→ EGNN performs BETTER")
    else:
        print("→ Models perform SIMILARLY")
    
    # Expected accuracy for reference
    expected_acc = 0.5 + f_pv / 4
    print(f"\nTheoretical maximum accuracy for f_pv={f_pv}: ~{expected_acc:.4f}")
    print("="*70)
    
    # Save results
    results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        save_results = {
            'config': results['config'],
            'egnn': {k: v for k, v in results['egnn'].items() if k != 'val_losses'},
            'frame_aligned': {k: v for k, v in results['frame_aligned'].items() if k != 'val_losses'},
            'timestamp': results['timestamp']
        }
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Create comparison plot
    plot_comparison(results, output_dir)
    
    return results


def plot_comparison(results, output_dir):
    """
    Create comparison plots for the two models.
    
    Args:
        results: Results dictionary from compare_models
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy comparison bar chart
    ax1 = axes[0]
    models = ['EGNN', 'Frame-Aligned GNN']
    means = [results['egnn']['mean_accuracy'], results['frame_aligned']['mean_accuracy']]
    stds = [results['egnn']['std_accuracy'], results['frame_aligned']['std_accuracy']]
    
    x = np.arange(len(models))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add theoretical maximum line
    f_pv = results['config']['f_pv']
    expected_acc = 0.5 + f_pv / 4
    ax1.axhline(y=expected_acc, color='green', linestyle='--', alpha=0.7,
               label=f'Theoretical max (~{expected_acc:.3f})')
    ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Random (0.5)')
    
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title(f'Model Comparison\n(n={results["config"]["n_points"]}, f_pv={f_pv})', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.set_ylim([0.45, max(means) + max(stds) + 0.05])
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Training convergence for each seed
    ax2 = axes[1]
    
    colors = {'egnn': '#1f77b4', 'frame_aligned': '#ff7f0e'}
    
    for model_name, display_name in [('egnn', 'EGNN'), ('frame_aligned', 'Frame-Aligned')]:
        for seed_idx, val_losses in enumerate(results[model_name]['val_losses']):
            epochs = range(1, len(val_losses) + 1)
            alpha_val = 0.5 if seed_idx > 0 else 1.0
            label = display_name if seed_idx == 0 else None
            ax2.plot(epochs, val_losses, color=colors[model_name], alpha=alpha_val,
                    linewidth=1.5, label=label)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Training Convergence', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare EGNN and Frame-Aligned GNN models for parity violation detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--n-points', type=int, default=10000,
                        help='Number of training+validation points')
    parser.add_argument('--f-pv', type=float, default=0.05,
                        help='Fraction of parity-violating pairs')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of random seeds for averaging')
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Hidden dimension for both models')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    compare_models(
        n_points=args.n_points,
        f_pv=args.f_pv,
        n_seeds=args.n_seeds,
        n_epochs=args.n_epochs,
        hidden_dim=args.hidden_dim,
        early_stopping_patience=args.early_stopping_patience,
        verbose=not args.quiet,
        output_dir=args.output_dir
    )

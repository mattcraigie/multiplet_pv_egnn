# 3D Parity-Violating Classifier for Spin-2 Objects

This repository implements a parity violation detection classifier for spin-2 objects in 3D space, using graph neural network architectures.

## Overview

The goal is to detect **parity violation in angle–position correlations** using learnable, rotation-invariant models. The system works with:

- **Spin-2 objects**: Headless orientations where angle φ and angle φ+π represent the same orientation (like line segments or galaxy ellipticities)
- **3D positions**: x, y coordinates with z (line-of-sight) depth
- **Parity-violating correlations**: Angle differences that correlate with line-of-sight ordering

## Model Types

The repository supports two model architectures:

### 1. Frame-Aligned GNN (Recommended - Default)

A new architecture where each node maintains its own local coordinate frame and a set of learned 2D latent vectors. Key features:

- Messages are rotated into receiver's coordinate frame for orientation-aware reasoning
- Uses spin-2 rotations (2θ) for period-π symmetry
- Incorporates 3D distance, z-separation, and orientation features
- Better performance on parity violation detection tasks

### 2. EGNN (Original)

A minimal EGNN-like message-passing classifier with:
- Node embedding from spin-2 angle features (cos(2φ), sin(2φ))
- Edge embedding from 3D distance, delta_z, and sin(2Δφ)
- Mean pooling and MLP classification head

## Repository Structure

```
multiplet_pv_egnn/
├── data.py                    # Dataset generation for 3D spin-2 objects
├── model.py                   # Original EGNN model implementation
├── frame_aligned_model.py     # Frame-Aligned GNN model implementation
├── experiments/               # All experiment scripts
│   ├── basic_train.py        # Basic training with visualization
│   ├── grid_test.py          # Grid test for detection capability analysis
│   ├── threshold_search.py   # Binary search for detection thresholds
│   └── compare_models.py     # Model comparison experiment
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Basic Training

Run basic training with the default Frame-Aligned GNN model:

```bash
# Train with Frame-Aligned GNN (default)
python -m experiments.basic_train --mode main --f-pv 1.0

# Full experiment suite (main + control + statistical tests)
python -m experiments.basic_train --mode full

# With visualization
python -m experiments.basic_train --mode full --visualize

# Use EGNN model instead
python -m experiments.basic_train --mode main --model-type egnn
```

### Model Comparison

Compare EGNN and Frame-Aligned GNN models:

```bash
# Default: 10,000 points, f_pv=0.05, 3 seeds
python -m experiments.compare_models

# Custom parameters
python -m experiments.compare_models --n-points 10000 --f-pv 0.05 --n-seeds 5
```

### Grid Test

Run parameter grid test for detection capability analysis:

```bash
# Using default configuration
python -m experiments.grid_test

# With custom config
python -m experiments.grid_test --config grid_test_config.yaml

# Generate plots from existing results
python -m experiments.grid_test --plot-only grid_test_results/results.json
```

### Threshold Search

Find detection boundaries (minimum f_pv for detection at each dataset size):

```bash
# Run threshold search
python -m experiments.threshold_search

# With custom config
python -m experiments.threshold_search --config threshold_config.yaml

# Generate plot from existing results
python -m experiments.threshold_search --plot-only threshold_search_results/boundary_results.json
```

## Key Parameters

- `f_pv`: Fraction of pairs that are parity-violating (0.0 to 1.0)
- `alpha`: Parity violation angle offset (default 0.3 rad)
- `model_type`: 'frame_aligned' (default) or 'egnn'
- `hidden_dim`: Hidden dimension for the model
- `num_slots`: Number of latent slots (for frame_aligned)
- `num_hops`: Number of message passing hops (for frame_aligned)

## Expected Results

For f_pv=1.0 (all pairs parity-violating), the theoretical maximum accuracy is ~75% (Bayes optimal). The Frame-Aligned GNN typically achieves closer to this limit compared to the original EGNN.

## Theory

The classifier distinguishes between:
1. **Parity-violating samples**: Angle differences correlate with line-of-sight ordering
2. **Parity-symmetric samples**: Angles are randomly flipped to remove the parity signature

Detection confidence scales with:
- **Dataset size**: Larger datasets enable detection of weaker signals
- **f_pv**: Higher parity violation fraction means stronger signal
- **Model capacity**: Frame-Aligned GNN better captures the geometric structure

The expected accuracy follows: accuracy ≈ 0.5 + f_pv/4 (theoretical limit)


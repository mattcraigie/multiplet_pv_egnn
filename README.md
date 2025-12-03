# 2D Parity-Violating EGNN Toy Experiment — Implementation Instructions

This document provides **clear, code-free instructions** for a coding agent to implement a minimal 2D test of a **parity-sensitive EGNN classifier**.  
The goal is to detect **parity violation in angle–position correlations** using a learnable, rotation-invariant model.

---

# 1. Problem Summary

We want to implement a toy system in **2D** where:

- Each sample is a **graph** of 2–N points (start with N=2 for minimal test).
- Each point has:
  - A 2D **position**.
  - An **orientation angle** φ ∈ [0, 2π).

The **positions** are isotropic and homogeneous.  
The **angles** contain a controlled, synthetic **parity-violating pattern**.

We then:

1. Build **real** samples containing parity-violating angle correlations.
2. Build **parity-symmetrized** samples by randomly flipping φ → −φ.
3. Train a classifier (EGNN) to distinguish the two datasets.

A classifier accuracy significantly above 50% indicates parity violation.

---

# 2. Dataset Specification

## 2.1 Each sample is a 2-point graph
For minimal demonstration:

- Graph has **2 nodes**.
- This is sufficient because relative angle differences already encode parity.

## 2.2 Node positions
For each pair:

1. Sample a **pair center** uniformly inside a square box.
2. Sample a random **axis direction** θ (uniform in [0, 2π)).
3. Sample a **separation distance** r from a given range.
4. Place the two points symmetrically around the center along direction θ.

This guarantees:
- Translational homogeneity  
- Rotational isotropy  
- Identical spatial statistics for real vs symmetrized data  

No parity information is encoded in positions alone.

## 2.3 Node orientations (angles)
Define a **true parity-violating pattern** as follows:

1. Sample a base orientation φ₀ uniformly in [0, 2π).
2. Fix a constant α (e.g., α = 0.5 rad).
3. Assign angles:
   - φ₁ = φ₀ + α  
   - φ₂ = φ₀ − α  

Thus:
- The two galaxies always differ by a **signed** relative angle Δφ = −2α.
- This fixed sign is the parity-violating signature.
- Because φ₀ is uniform, there is **no preferred global orientation**, only local chirality.

This produces a globally isotropic but parity-violating dataset.

## 2.4 Parity-symmetrized dataset (control)
To create the matched control dataset:

For each real sample:

- With 50% probability, **flip all angles**:  
  φ → −φ (mod 2π)
- With 50% probability, leave the angles unchanged.

This ensures:
- Angle **marginals** match exactly.
- All spatial statistics match exactly.
- Only parity-odd structure differs.

This is the perfect contrast for a classifier.

---

# 3. Graph Representation

Each graph contains:

- Node features:
  - cos(φ)
  - sin(φ)
- Edge features:
  - Pairwise distance d₁₂  
  - sin(Δφ) = sin(φ₂ − φ₁)

Notes:

- **cos(Δφ)** is parity-even → provides no discrimination → may be omitted.
- **sin(Δφ)** is parity-odd → essential for detection.

Because we use **distances**, the graph is **rotation-invariant** by construction.  
Thus **no rotation augmentation** is required.

---

# 4. Model Specification (Conceptual)

Implement a **minimal EGNN-like message-passing classifier** with:

1. **No coordinate updates** (distances already give rotation invariance).
2. **Node embedding** from angle features.
3. **Edge embedding** from:
   - distance
   - parity-odd angle feature sin(Δφ)

4. **Message passing layers**:
   - Messages depend on node features and edge features.
   - Node updates aggregate messages via summation.

5. **Graph readout**:
   - Mean-pool all node embeddings.
   - Feed pooled vector into an MLP producing a single **logit**.

This logit represents:
- 1 → real parity-violating  
- 0 → symmetrized parity-symmetric  

Even a very small EGNN (~1–2 layers, 16 hidden dims) is enough.

---

# 5. Training Procedure

## 5.1 Construct training, validation, test splits
Each split contains equal numbers of:
- Real samples (label = 1)
- Symmetrized samples (label = 0)

## 5.2 Training
Use:

- Binary cross-entropy loss on the classifier logit.
- Adam optimizer.
- Small batch size (~64–128).

Training should converge within ~20 epochs.

## 5.3 Evaluation metric
Compute:

- Classification accuracy  
- AUC (optional)

Interpretation:

- Accuracy ≈ 0.5 → **no detectable parity violation**  
- Accuracy > 0.5 → **detected parity violation**  

The parity-symmetric control dataset should always give ≈0.5.

---

# 6. Randomized Statistical Test

To make detection robust:

1. Run the entire training procedure across multiple random seeds.
2. Collect test accuracies.
3. Compare mean accuracy against 0.5.

A consistent uplift (e.g., 0.70 ± 0.03) confirms parity violation.

To ensure the method is valid:

- Replace the real dataset with a **purely symmetric angle generator** (angles completely random).
- Ensure accuracy drops back to ~0.5.

---

# 7. Summary of What the Agent Must Implement

## Data generation
- Generate isotropic 2D point pairs.
- Assign angles with fixed-signed relative offsets.
- Build symmetrized dataset by φ → −φ flips.

## Graph construction
- Node features: (cos φ, sin φ)
- Edge features: (distance, sin(Δφ))

## Model pipeline
- A minimal EGNN-like message-passing network.
- No coordinate updates required.
- Mean pooling → MLP → single logit.

## Training
- Binary classification: real vs symmetrized.
- BCE loss.
- Accuracy metric.

## Validation
- Should detect parity when α > 0.
- Should fail (accuracy → 0.5) for parity-symmetric control.

---

# 8. Optional Extensions (future improvements)
The agent may add these later if desired:

- Larger graphs (N > 2).
- Message passing with learned edge weights.
- Coordinate updates for a full EGNN.
- Explainability (saliency → per-node parity score).
- Sweep α values to measure detection sensitivity curve.

---

This completes the high-level implementation plan.

The coding agent should now:

1. Follow the dataset instructions exactly.  
2. Build the graph structure as described.  
3. Implement a minimal EGNN classifier.  
4. Train and evaluate the model.  
5. Verify that it detects parity violation correctly.


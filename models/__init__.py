"""
Models module for 3D Parity Violation Detection.

This module contains:
- ParityViolationEGNN: EGNN-like message-passing classifier
- MultiHopParityViolationEGNN: EGNN for variable-size graphs with multi-hop message passing
- FrameAlignedPVClassifier: Frame-aligned classifier with latent slots (recommended)
- MultiHopFrameAlignedPVClassifier: Frame-aligned classifier for multi-hop graphs
"""

from models.model import (
    MessagePassingLayer,
    ParityViolationEGNN,
    MultiHopMessagePassingLayer,
    MultiHopParityViolationEGNN,
)

from models.frame_aligned_model import (
    FrameAlignedGNNLayer3D,
    FrameAlignedGNN3D,
    FrameAlignedPVClassifier,
    MultiHopFrameAlignedPVClassifier,
    # Legacy aliases
    FrameAlignedGNNLayer,
    FrameAlignedGNN,
)

__all__ = [
    # EGNN models
    'MessagePassingLayer',
    'ParityViolationEGNN',
    'MultiHopMessagePassingLayer',
    'MultiHopParityViolationEGNN',
    # Frame-aligned models
    'FrameAlignedGNNLayer3D',
    'FrameAlignedGNN3D',
    'FrameAlignedPVClassifier',
    'MultiHopFrameAlignedPVClassifier',
    # Legacy
    'FrameAlignedGNNLayer',
    'FrameAlignedGNN',
]

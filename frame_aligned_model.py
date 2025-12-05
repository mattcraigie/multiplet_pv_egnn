"""
Frame-Aligned GNN for 3D Parity Violation Detection with Spin-2 Objects.

This model performs message passing where every node maintains its own local coordinate frame 
and a set of N learned 2D vectors (latent slots) that represent structured information about 
its surroundings. When a neighbor sends a message, the latent vectors from that neighbor are 
rotated into the receiver's coordinate frame so that all information is expressed consistently 
relative to the receiver's orientation.

Adapted for:
1. 3D positions: x, y positions with z (line-of-sight) depth
2. Spin-2 objects: Orientations have period π, so we use 2θ for rotations

The key insight for spin-2: Since orientations repeat every π radians (not 2π), we must
double the angles when rotating latent vectors to ensure proper periodicity.
"""

import torch
import torch.nn as nn


class FrameAlignedGNNLayer3D(nn.Module):
    """
    Frame-aligned message passing layer with Nx2 vector slots per node.
    Adapted for 3D positions with spin-2 orientations.

    Each node i has:
      - position x[i] in R^3        (x, y, z global coordinates)
      - orientation theta[i] in [0, π)  (radians, spin-2 orientation in x-y plane)
      - latent H[i] in R^{num_slots x 2} (vectors in node i's local frame)

    For each directed edge j -> i:
      - Rotate H[j] from node j's frame into node i's frame (using 2θ for spin-2).
      - Compute geometric features in i's frame:
          r_3d, r_xy, Δz, Δphi_ij, Δtheta_ij
        encoded with sin/cos for angular features.
      - Combine rotated latent + geom features via per-slot MLP.
      - Aggregate messages over incoming edges (sum).

    Shapes:
      x:          [num_nodes, 3]    (x, y, z positions)
      theta:      [num_nodes]       (orientation angles in [0, π))
      H:          [num_nodes, num_slots, 2]
      edge_index: [2, num_edges]   (row 0 = i, row 1 = j; messages j -> i)

    Output:
      H_new:      [num_nodes, num_slots, 2]
    """

    def __init__(self, num_slots: int, hidden_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim

        # Input to slot MLP:
        #   2 (slot vector in i-frame) +
        #   8 (geom features: r_3d, r_xy, Δz, sinΔφ, cosΔφ, sin2Δθ, cos2Δθ, sign(Δz)*sin2Δθ)
        in_dim = 2 + 8

        self.slot_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        x: torch.Tensor,          # [N, 3]
        theta: torch.Tensor,      # [N]
        H: torch.Tensor,          # [N, S, 2]
        edge_index: torch.Tensor  # [2, E] with edges j -> i
    ) -> torch.Tensor:
        N, S = H.shape[0], H.shape[1]
        assert S == self.num_slots

        i_idx = edge_index[0]  # targets
        j_idx = edge_index[1]  # sources
        E = i_idx.shape[0]

        # --- 1. Relative geometry in global frame ---
        dx = x[j_idx] - x[i_idx]                      # [E, 3]
        dx_xy = dx[:, :2]                             # [E, 2] x-y plane
        dz = dx[:, 2]                                 # [E] line-of-sight separation
        
        r_3d = torch.norm(dx, dim=-1, keepdim=True)   # [E, 1] 3D distance
        r_xy = torch.norm(dx_xy, dim=-1, keepdim=True)  # [E, 1] x-y plane distance
        
        # Direction angle in x-y plane from node i to node j
        phi_ij = torch.atan2(dx_xy[:, 1], dx_xy[:, 0])  # [E]

        theta_i = theta[i_idx]  # [E]
        theta_j = theta[j_idx]  # [E]

        # Δphi_ij = direction of (j - i) minus i's orientation
        delta_phi = phi_ij - theta_i          # [E]
        
        # Δtheta_ij = orientation difference (for spin-2, use 2θ for proper period)
        # This is the key adaptation for spin-2
        delta_theta = theta_j - theta_i       # [E]

        # Geometric features with sin/cos encoding
        # For spin-2, we use sin(2Δθ) and cos(2Δθ) which have period π
        geom_feat = torch.stack(
            [
                r_3d.squeeze(-1),                    # [E] 3D distance
                r_xy.squeeze(-1),                    # [E] x-y distance
                dz,                                  # [E] z-separation (signed)
                torch.sin(delta_phi),               # [E]
                torch.cos(delta_phi),               # [E]
                torch.sin(2 * delta_theta),         # [E] spin-2: 2θ
                torch.cos(2 * delta_theta),         # [E] spin-2: 2θ
                torch.sign(dz) * torch.sin(2 * delta_theta),  # [E] PV-sensitive feature
            ],
            dim=-1,
        )  # [E, 8]

        # --- 2. Rotate neighbor latent from j's frame into i's frame ---
        # For spin-2, we rotate by 2(theta_j - theta_i) to respect period-π symmetry
        # This ensures that rotating by π (which is equivalent to identity for spin-2)
        # results in rotation by 2π (which is identity for 2D vectors)
        angle_ji = 2 * (theta_j - theta_i)    # [E] - doubled for spin-2
        cos_a = torch.cos(angle_ji)           # [E]
        sin_a = torch.sin(angle_ji)           # [E]

        H_j = H[j_idx]                        # [E, S, 2]

        # Rotate each 2D slot vector by angle_ji
        # v' = R(Δθ) v, with R = [[cos, -sin],[sin, cos]]
        v_x = H_j[..., 0]  # [E, S]
        v_y = H_j[..., 1]  # [E, S]

        v_x_rot = cos_a.unsqueeze(-1) * v_x - sin_a.unsqueeze(-1) * v_y
        v_y_rot = sin_a.unsqueeze(-1) * v_x + cos_a.unsqueeze(-1) * v_y

        H_j_in_i = torch.stack([v_x_rot, v_y_rot], dim=-1)  # [E, S, 2]

        # --- 3. Build per-slot input: [v_in_i_frame, geom_feat] ---
        geom_expanded = geom_feat.unsqueeze(1).expand(E, S, -1)  # [E, S, 8]
        slot_input = torch.cat([H_j_in_i, geom_expanded], dim=-1)  # [E, S, 10]

        slot_input_flat = slot_input.reshape(E * S, -1)  # [E*S, 10]

        # --- 4. Slot-wise MLP to produce messages ---
        msg_flat = self.slot_mlp(slot_input_flat)  # [E*S, 2]
        M = msg_flat.view(E, S, 2)                 # [E, S, 2]

        # --- 5. Aggregate messages over incoming edges j -> i ---
        H_new = torch.zeros_like(H)  # [N, S, 2]
        H_new.index_add_(0, i_idx, M)

        return H_new


class FrameAlignedGNN3D(nn.Module):
    """
    Full frame-aligned GNN with num_hops message passing steps for 3D spin-2 objects.

    - num_hops includes the first "special" hop:
      we initialize H^{(0)} = 0 and run num_hops times.

    - All layers share the same computation pattern, but have
      separate parameters (ModuleList), so the first hop can
      learn a different function than later hops.

    Usage:
        model = FrameAlignedGNN3D(
            num_slots=8,
            hidden_dim=64,
            num_hops=3
        )

        H_out = model(x, theta, edge_index)
        # H_out: [num_nodes, num_slots, 2]
    """

    def __init__(self, num_slots: int, hidden_dim: int, num_hops: int):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops

        self.layers = nn.ModuleList([
            FrameAlignedGNNLayer3D(num_slots=num_slots, hidden_dim=hidden_dim)
            for _ in range(num_hops)
        ])

    def forward(
        self,
        x: torch.Tensor,          # [N, 3]
        theta: torch.Tensor,      # [N]
        edge_index: torch.Tensor, # [2, E]
        H0: torch.Tensor = None   # optional [N, S, 2], default zeros
    ) -> torch.Tensor:
        N = x.shape[0]
        S = self.num_slots

        if H0 is None:
            # First "special" step: zero latent, purely geometric embedding
            H = torch.zeros(N, S, 2, device=x.device, dtype=x.dtype)
        else:
            H = H0
            assert H.shape == (N, S, 2)

        for layer in self.layers:
            H = layer(x, theta, H, edge_index)

        # H is in each node's local frame after num_hops steps
        return H


class FrameAlignedPVClassifier(nn.Module):
    """
    Frame-aligned classifier for 3D parity violation detection with spin-2 objects.
    
    This wraps the FrameAlignedGNN3D with:
    1. An initial node embedding layer
    2. A readout layer that aggregates slot vectors
    3. A classification head
    
    Architecture:
    1. Frame-aligned message passing to build latent representations
    2. Mean pool over latent slots (after taking norm to make invariant)
    3. MLP classifier producing single logit
    """
    
    def __init__(
        self,
        num_slots: int = 8,
        hidden_dim: int = 32,
        num_hops: int = 2,
        readout_dim: int = 32
    ):
        """
        Initialize the classifier.
        
        Args:
            num_slots: Number of latent vector slots per node
            hidden_dim: Hidden dimension for message passing MLPs
            num_hops: Number of message passing iterations
            readout_dim: Dimension for the readout MLP
        """
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        
        # Frame-aligned GNN backbone
        self.gnn = FrameAlignedGNN3D(
            num_slots=num_slots,
            hidden_dim=hidden_dim,
            num_hops=num_hops
        )
        
        # Readout: process each slot's norm and pool
        # Slot vectors are 2D, so we compute their norm to get rotation-invariant features
        # Then we process with an MLP
        self.slot_readout = nn.Sequential(
            nn.Linear(num_slots, readout_dim),  # num_slots norms
            nn.SiLU(),
            nn.Linear(readout_dim, readout_dim)
        )
        
        # Classification head: from node features to graph logit
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, readout_dim),
            nn.SiLU(),
            nn.Linear(readout_dim, 1)
        )
    
    def forward(
        self,
        positions: torch.Tensor,   # [batch, 2, 3] - 3D positions of 2 nodes per graph
        angles: torch.Tensor,      # [batch, 2] - orientation angles in [0, π)
    ) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            positions: (batch, 2, 3) tensor with 3D positions for each node pair
            angles: (batch, 2) tensor with angles in [0, π) for each node
            
        Returns:
            logits: (batch,) tensor with classification logits
        """
        batch_size = positions.shape[0]
        device = positions.device
        
        # Flatten batch to single graph for message passing
        # We treat each pair as a separate 2-node graph
        # positions: [B, 2, 3] -> x: [B*2, 3]
        x = positions.view(-1, 3)  # [B*2, 3]
        
        # angles: [B, 2] -> theta: [B*2]
        theta = angles.view(-1)  # [B*2]
        
        # Create edge index for bidirectional edges in each 2-node graph efficiently
        # For graph g: nodes are (2*g, 2*g+1)
        # Edges: (2*g) -> (2*g+1), (2*g+1) -> (2*g)
        g = torch.arange(batch_size, device=device)
        n0 = 2 * g      # first nodes: 0, 2, 4, ...
        n1 = 2 * g + 1  # second nodes: 1, 3, 5, ...
        # Targets: [n0_0, n1_0, n0_1, n1_1, ...] interleaved
        i_idx = torch.stack([n0, n1], dim=1).view(-1)  # [2*B]
        # Sources: [n1_0, n0_0, n1_1, n0_1, ...] interleaved  
        j_idx = torch.stack([n1, n0], dim=1).view(-1)  # [2*B]
        edge_index = torch.stack([i_idx, j_idx], dim=0)  # [2, 2*B]
        
        # Run frame-aligned message passing
        H = self.gnn(x, theta, edge_index)  # [B*2, num_slots, 2]
        
        # Compute rotation-invariant features: norm of each slot vector
        slot_norms = torch.norm(H, dim=-1)  # [B*2, num_slots]
        
        # Reshape to [B, 2, num_slots]
        slot_norms = slot_norms.view(batch_size, 2, self.num_slots)
        
        # Mean pool over nodes to get graph-level features
        graph_features = slot_norms.mean(dim=1)  # [B, num_slots]
        
        # Readout MLP
        readout = self.slot_readout(graph_features)  # [B, readout_dim]
        
        # Classification
        logits = self.classifier(readout).squeeze(-1)  # [B]
        
        return logits


class MultiHopFrameAlignedPVClassifier(nn.Module):
    """
    Frame-aligned classifier for multi-hop parity violation detection with spin-2 objects.
    
    This classifier works with variable-size graphs where each graph has N nodes
    and edges defined by a k-NN or radius graph.
    
    Architecture:
    1. Frame-aligned message passing to build latent representations
    2. Mean pool over all nodes (after taking norm to make rotation-invariant)
    3. MLP classifier producing single logit
    """
    
    def __init__(
        self,
        num_slots: int = 8,
        hidden_dim: int = 32,
        num_hops: int = 3,
        readout_dim: int = 32
    ):
        """
        Initialize the multi-hop classifier.
        
        Args:
            num_slots: Number of latent vector slots per node
            hidden_dim: Hidden dimension for message passing MLPs
            num_hops: Number of message passing iterations
            readout_dim: Dimension for the readout MLP
        """
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        
        # Frame-aligned GNN backbone
        self.gnn = FrameAlignedGNN3D(
            num_slots=num_slots,
            hidden_dim=hidden_dim,
            num_hops=num_hops
        )
        
        # Readout: process each slot's norm and pool
        self.slot_readout = nn.Sequential(
            nn.Linear(num_slots, readout_dim),
            nn.SiLU(),
            nn.Linear(readout_dim, readout_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, readout_dim),
            nn.SiLU(),
            nn.Linear(readout_dim, 1)
        )
    
    def forward(
        self,
        positions: torch.Tensor,   # [N, 3] - 3D positions (batched together)
        angles: torch.Tensor,      # [N] - orientation angles in [0, π)
        edge_index: torch.Tensor,  # [2, E] - edge connectivity
        batch: torch.Tensor        # [N] - batch assignment for each node
    ) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            positions: (N, 3) tensor with 3D positions for all nodes
            angles: (N,) tensor with angles in [0, π) for each node
            edge_index: (2, E) tensor with graph connectivity
            batch: (N,) tensor indicating which graph each node belongs to
            
        Returns:
            logits: (batch_size,) tensor with classification logits
        """
        # Run frame-aligned message passing
        H = self.gnn(positions, angles, edge_index)  # [N, num_slots, 2]
        
        # Compute rotation-invariant features: norm of each slot vector
        slot_norms = torch.norm(H, dim=-1)  # [N, num_slots]
        
        # Global mean pooling over nodes in each graph
        batch_size = batch.max().item() + 1
        graph_features = torch.zeros(batch_size, self.num_slots, device=positions.device)
        node_counts = torch.zeros(batch_size, device=positions.device)
        
        # Scatter add for pooling
        graph_features.index_add_(0, batch, slot_norms)
        node_counts.index_add_(0, batch, torch.ones(batch.shape[0], device=positions.device))
        
        # Mean pooling
        graph_features = graph_features / node_counts.unsqueeze(-1).clamp(min=1)
        
        # Readout MLP
        readout = self.slot_readout(graph_features)  # [batch_size, readout_dim]
        
        # Classification
        logits = self.classifier(readout).squeeze(-1)  # [batch_size]
        
        return logits


# Legacy aliases for compatibility
FrameAlignedGNNLayer = FrameAlignedGNNLayer3D
FrameAlignedGNN = FrameAlignedGNN3D

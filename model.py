"""
EGNN-like message-passing classifier for 3D parity violation detection with spin-2 objects.

This implements a minimal EGNN-like architecture:
- No coordinate updates (distances give rotation invariance)
- Node embedding from spin-2 angle features (cos(2φ), sin(2φ))
- Edge embedding from 3D distance, delta_z, and sin(2Δφ) for spin-2
- Message passing with summation aggregation
- Mean pooling → MLP → single logit
"""

import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    """
    Single message passing layer for the EGNN.
    
    Messages depend on:
    - Source node features
    - Target node features
    - Edge features (distance, sin(Δφ))
    
    Node updates aggregate messages via summation.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        """
        Initialize the message passing layer.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for the message MLP
        """
        super().__init__()
        
        # Message MLP: takes concatenated [node_i, node_j, edge_ij]
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Node update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, node_features, edge_features):
        """
        Forward pass through the message passing layer.
        
        For a 2-node graph, we compute messages between both pairs
        and aggregate by summation.
        
        Args:
            node_features: (batch, 2, node_dim) tensor
            edge_features: (batch, edge_dim) tensor
            
        Returns:
            Updated node features (batch, 2, node_dim)
        """
        batch_size = node_features.shape[0]
        
        # For 2-node graph:
        # Node 0 receives message from node 1
        # Node 1 receives message from node 0
        
        node_0 = node_features[:, 0, :]  # (batch, node_dim)
        node_1 = node_features[:, 1, :]  # (batch, node_dim)
        
        # Message from 1 to 0
        msg_input_01 = torch.cat([node_0, node_1, edge_features], dim=-1)
        msg_01 = self.message_mlp(msg_input_01)  # (batch, node_dim)
        
        # Message from 0 to 1
        msg_input_10 = torch.cat([node_1, node_0, edge_features], dim=-1)
        msg_10 = self.message_mlp(msg_input_10)  # (batch, node_dim)
        
        # Update nodes
        update_0 = self.update_mlp(torch.cat([node_0, msg_01], dim=-1))
        update_1 = self.update_mlp(torch.cat([node_1, msg_10], dim=-1))
        
        # Stack back into (batch, 2, node_dim)
        updated_nodes = torch.stack([update_0, update_1], dim=1)
        
        # Residual connection
        return node_features + updated_nodes


class ParityViolationEGNN(nn.Module):
    """
    EGNN-like classifier for 3D parity violation detection with spin-2 objects.
    
    Architecture:
    1. Node embedding from spin-2 angle features (cos(2φ), sin(2φ))
    2. Edge embedding from 3D distance, delta_z, and sin(2Δφ) for spin-2
    3. Message passing layers
    4. Mean pooling over nodes
    5. MLP classifier producing single logit
    """
    
    def __init__(
        self,
        node_input_dim: int = 2,
        edge_input_dim: int = 3,
        hidden_dim: int = 16,
        n_layers: int = 2
    ):
        """
        Initialize the EGNN classifier.
        
        Args:
            node_input_dim: Input dimension for node features (default 2 for cos(2φ)/sin(2φ))
            edge_input_dim: Input dimension for edge features (default 3 for distance_3d, delta_z, sin_2delta_phi)
            hidden_dim: Hidden dimension throughout the network
            n_layers: Number of message passing layers
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Classification head: mean pool → MLP → logit
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_features, edge_distance_3d, edge_delta_z, edge_sin_2delta_phi):
        """
        Forward pass through the EGNN classifier.
        
        Args:
            node_features: (batch, 2, 2) tensor with (cos(2φ), sin(2φ)) for each node
            edge_distance_3d: (batch,) tensor with 3D pairwise distances
            edge_delta_z: (batch,) tensor with signed line-of-sight separations
            edge_sin_2delta_phi: (batch,) tensor with sin(2Δφ) values for spin-2
            
        Returns:
            logits: (batch,) tensor with classification logits
        """
        batch_size = node_features.shape[0]
        
        # Embed nodes
        # Reshape to (batch * 2, node_input_dim) for batch processing
        node_flat = node_features.view(-1, node_features.shape[-1])
        node_embed = self.node_embed(node_flat)
        node_embed = node_embed.view(batch_size, 2, self.hidden_dim)
        
        # Embed edges (3 features: distance_3d, delta_z, sin_2delta_phi)
        edge_input = torch.stack([edge_distance_3d, edge_delta_z, edge_sin_2delta_phi], dim=-1)
        edge_embed = self.edge_embed(edge_input)  # (batch, hidden_dim)
        
        # Message passing
        h = node_embed
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_embed)
        
        # Mean pooling over nodes
        graph_embed = h.mean(dim=1)  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(graph_embed).squeeze(-1)  # (batch,)
        
        return logits


class MultiHopMessagePassingLayer(nn.Module):
    """
    Message passing layer for variable-size graphs.
    
    Messages depend on:
    - Source node features
    - Target node features
    - Edge features (distance, delta_z)
    
    Node updates aggregate messages via summation.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        """
        Initialize the message passing layer.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for the message MLP
        """
        super().__init__()
        
        # Message MLP: takes concatenated [node_i, node_j, edge_ij]
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Node update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, node_features, edge_index, edge_features):
        """
        Forward pass through the message passing layer.
        
        Args:
            node_features: (N, node_dim) tensor
            edge_index: (2, E) tensor with [targets, sources]
            edge_features: (E, edge_dim) tensor
            
        Returns:
            Updated node features (N, node_dim)
        """
        n_nodes = node_features.shape[0]
        
        i_idx = edge_index[0]  # targets
        j_idx = edge_index[1]  # sources
        
        # Get source and target node features
        node_i = node_features[i_idx]  # (E, node_dim)
        node_j = node_features[j_idx]  # (E, node_dim)
        
        # Compute messages
        msg_input = torch.cat([node_i, node_j, edge_features], dim=-1)  # (E, 2*node_dim + edge_dim)
        messages = self.message_mlp(msg_input)  # (E, node_dim)
        
        # Aggregate messages by summing over incoming edges
        aggregated = torch.zeros_like(node_features)  # (N, node_dim)
        aggregated.index_add_(0, i_idx, messages)
        
        # Update nodes
        update_input = torch.cat([node_features, aggregated], dim=-1)
        updated = self.update_mlp(update_input)
        
        # Residual connection
        return node_features + updated


class MultiHopParityViolationEGNN(nn.Module):
    """
    EGNN-like classifier for multi-hop parity violation detection with spin-2 objects.
    
    This classifier works with variable-size graphs where each graph has N nodes
    and edges defined by a k-NN or radius graph.
    
    Architecture:
    1. Node embedding from spin-2 angle features (cos(2φ), sin(2φ))
    2. Edge embedding computed from positions (distance, delta_z)
    3. Message passing layers with general graphs
    4. Mean pooling over nodes
    5. MLP classifier producing single logit
    """
    
    def __init__(
        self,
        node_input_dim: int = 2,
        hidden_dim: int = 32,
        n_layers: int = 3
    ):
        """
        Initialize the multi-hop EGNN classifier.
        
        Args:
            node_input_dim: Input dimension for node features (default 2 for cos(2φ)/sin(2φ))
            hidden_dim: Hidden dimension throughout the network
            n_layers: Number of message passing layers
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge embedding input: distance_3d, distance_xy, delta_z
        edge_input_dim = 3
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MultiHopMessagePassingLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Classification head: mean pool → MLP → logit
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        positions: torch.Tensor,   # [N, 3] - 3D positions
        node_features: torch.Tensor,  # [N, 2] - (cos(2φ), sin(2φ))
        edge_index: torch.Tensor,  # [2, E] - edge connectivity
        batch: torch.Tensor        # [N] - batch assignment for each node
    ) -> torch.Tensor:
        """
        Forward pass through the multi-hop EGNN classifier.
        
        Args:
            positions: (N, 3) tensor with 3D positions for all nodes
            node_features: (N, 2) tensor with node features
            edge_index: (2, E) tensor with graph connectivity
            batch: (N,) tensor indicating which graph each node belongs to
            
        Returns:
            logits: (batch_size,) tensor with classification logits
        """
        # Embed nodes
        h = self.node_embed(node_features)  # (N, hidden_dim)
        
        # Compute edge features from positions
        i_idx = edge_index[0]  # targets
        j_idx = edge_index[1]  # sources
        
        dx = positions[j_idx] - positions[i_idx]  # (E, 3)
        distance_3d = torch.norm(dx, dim=-1, keepdim=True)  # (E, 1)
        distance_xy = torch.norm(dx[:, :2], dim=-1, keepdim=True)  # (E, 1)
        delta_z = dx[:, 2:3]  # (E, 1)
        
        edge_input = torch.cat([distance_3d, distance_xy, delta_z], dim=-1)  # (E, 3)
        edge_features = self.edge_embed(edge_input)  # (E, hidden_dim)
        
        # Message passing
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_features)
        
        # Global mean pooling over nodes in each graph
        batch_size = batch.max().item() + 1
        graph_embed = torch.zeros(batch_size, self.hidden_dim, device=positions.device)
        node_counts = torch.zeros(batch_size, device=positions.device)
        
        # Scatter add for pooling
        graph_embed.index_add_(0, batch, h)
        node_counts.index_add_(0, batch, torch.ones(batch.shape[0], device=positions.device))
        
        # Mean pooling
        graph_embed = graph_embed / node_counts.unsqueeze(-1).clamp(min=1)
        
        # Classification
        logits = self.classifier(graph_embed).squeeze(-1)  # (batch_size,)
        
        return logits

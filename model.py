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

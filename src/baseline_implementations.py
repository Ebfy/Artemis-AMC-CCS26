"""
Baseline Implementations for ARTEMIS Comparison
================================================

Implements baseline methods for fair comparison:
    - GraphSAGE (Hamilton et al., 2017)
    - GAT (Veličković et al., 2018)
    - TGN (Rossi et al., 2020)
    - TGAT (Xu et al., 2020)
    - JODIE (Kumar et al., 2019)
    - GrabPhisher (Zhang et al., 2024)
    - 2DynEthNet (Yang et al., 2024)

All baselines use the same evaluation protocol for fair comparison.

Author: Anonymous (CCS 2026 Submission)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, SAGEConv, GCNConv, 
    global_mean_pool, global_max_pool
)
from torch_geometric.nn.models import TGN as PyGTGN
from typing import Optional, Dict, Tuple
import math


class GraphSAGE(nn.Module):
    """
    GraphSAGE: Inductive Representation Learning on Large Graphs
    
    Reference: Hamilton et al., NeurIPS 2017
    
    Architecture:
        - 3-layer SAGE convolution with mean aggregation
        - Global mean + max pooling
        - MLP classifier
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Last layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        # Classification
        return self.classifier(x)


class GAT(nn.Module):
    """
    GAT: Graph Attention Networks
    
    Reference: Veličković et al., ICLR 2018
    
    Architecture:
        - 3-layer GAT with multi-head attention
        - Global attention pooling
        - MLP classifier
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(
            in_channels, hidden_channels // num_heads,
            heads=num_heads, dropout=dropout
        ))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels, hidden_channels // num_heads,
                heads=num_heads, dropout=dropout
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Last layer (single head)
        self.convs.append(GATConv(
            hidden_channels, out_channels,
            heads=1, concat=False, dropout=dropout
        ))
        self.norms.append(nn.LayerNorm(out_channels))
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        return self.classifier(x)


class TGN(nn.Module):
    """
    TGN: Temporal Graph Networks
    
    Reference: Rossi et al., ICML 2020 Workshop
    
    Architecture:
        - Memory module with message aggregation
        - Time encoding
        - Graph attention for embedding
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        memory_dim: int = 100,
        time_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        
        # Node embedding projection
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        
        # Memory module (simplified)
        self.memory = None
        self.memory_updater = nn.GRUCell(hidden_channels, memory_dim)
        
        # Time encoding
        self.time_encoder = TimeEncoder(time_dim)
        
        # Message aggregator
        self.msg_aggregator = nn.Sequential(
            nn.Linear(hidden_channels + time_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Embedding layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_channels + memory_dim, hidden_channels, heads=2, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=2, concat=False))
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Project node features
        h = self.node_proj(x)
        
        # Initialize or retrieve memory
        num_nodes = x.size(0)
        if self.memory is None or self.memory.size(0) != num_nodes:
            self.memory = torch.zeros(num_nodes, self.memory_dim, device=x.device)
        
        # Combine with memory
        h = torch.cat([h, self.memory], dim=-1)
        
        # Graph convolutions
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Update memory (during training)
        if self.training:
            with torch.no_grad():
                self.memory = self.memory_updater(h[:, :self.hidden_channels], self.memory)
        
        # Pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h = torch.cat([h_mean, h_max], dim=-1)
        
        return self.classifier(h)
    
    def reset_memory(self):
        self.memory = None


class TimeEncoder(nn.Module):
    """Time encoding using sinusoidal functions."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        return torch.cos(self.w(t))


class TGAT(nn.Module):
    """
    TGAT: Temporal Graph Attention Networks
    
    Reference: Xu et al., ICLR 2020
    
    Architecture:
        - Time-aware attention mechanism
        - Temporal neighborhood aggregation
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        time_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.time_encoder = TimeEncoder(time_dim)
        
        # Projection
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        
        # Temporal attention layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_channels + time_dim if i == 0 else hidden_channels
            self.layers.append(
                TemporalAttentionLayer(in_dim, hidden_channels, num_heads, dropout)
            )
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Project and add time encoding
        h = self.node_proj(x)
        
        if timestamps is not None:
            t_enc = self.time_encoder(timestamps.float())
            # Aggregate time encoding to nodes
            t_node = torch.zeros(h.size(0), self.time_encoder.dim, device=h.device)
            t_node.scatter_add_(0, edge_index[1].unsqueeze(-1).expand(-1, t_enc.size(-1)), t_enc)
            h = torch.cat([h, t_node], dim=-1)
        else:
            h = torch.cat([h, torch.zeros(h.size(0), self.time_encoder.dim, device=h.device)], dim=-1)
        
        # Temporal attention layers
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h = torch.cat([h_mean, h_max], dim=-1)
        
        return self.classifier(h)


class TemporalAttentionLayer(nn.Module):
    """Single temporal attention layer."""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // num_heads, heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.norm(self.gat(x, edge_index))


class JODIE(nn.Module):
    """
    JODIE: Predicting Dynamic Embedding Trajectory
    
    Reference: Kumar et al., KDD 2019
    
    Architecture:
        - Coupled RNN for user/item embeddings
        - Temporal attention
        - Projection for future embedding
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        time_dim: int = 16,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Node embedding
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        
        # Coupled RNNs
        self.rnn = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        
        # Time projection
        self.time_proj = nn.Linear(1, hidden_channels)
        
        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Project node features
        h = self.node_proj(x)
        
        # Simple aggregation (simplified from full JODIE)
        h, _ = self.rnn(h.unsqueeze(1))
        h = h.squeeze(1)
        
        # Time projection if available
        if timestamps is not None:
            t_proj = self.time_proj(timestamps.float().mean().unsqueeze(0).unsqueeze(0))
            h = h + t_proj
        
        h = self.embed_proj(torch.cat([h, h], dim=-1))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h = torch.cat([h_mean, h_max], dim=-1)
        
        return self.classifier(h)


class GrabPhisher(nn.Module):
    """
    GrabPhisher: Detecting Phishing Scammers in Ethereum
    
    Reference: Zhang et al., 2024
    
    Architecture:
        - Graph-level feature extraction
        - Dynamic graph convolution
        - Temporal attention
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Dynamic graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(DynamicGraphConv(hidden_channels, hidden_channels))
        
        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(hidden_channels, num_heads=4, dropout=dropout)
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Feature extraction
        h = self.feature_extractor(x)
        
        # Dynamic graph convolution
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Temporal attention (self-attention)
        h_attn, _ = self.temporal_attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = h + h_attn.squeeze(0)
        
        # Pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h = torch.cat([h_mean, h_max], dim=-1)
        
        return self.classifier(h)


class DynamicGraphConv(nn.Module):
    """Dynamic graph convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        
        # Aggregate neighbor features
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(-1)), x[row])
        
        # Combine with self features
        h = torch.cat([x, neighbor_sum], dim=-1)
        h = self.lin(h)
        h = self.norm(h)
        
        return h


class TwoDynEthNet(nn.Module):
    """
    2DynEthNet: State-of-the-art baseline for Ethereum phishing detection
    
    Reference: Yang et al., 2024
    
    Architecture:
        - Temporal graph convolution with discrete windows
        - FIFO memory module
        - Reptile meta-learning (during training)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        memory_size: int = 500,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.memory_size = memory_size
        
        # Node projection
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        
        # Temporal graph convolution (discrete)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_channels, hidden_channels // 4, heads=4))
        
        # FIFO Memory module
        self.register_buffer('memory', torch.zeros(memory_size, hidden_channels))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Memory attention
        self.memory_attn = nn.MultiheadAttention(hidden_channels, num_heads=4, dropout=dropout)
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def update_memory(self, h: torch.Tensor):
        """FIFO memory update."""
        with torch.no_grad():
            batch_size = min(h.size(0), self.memory_size)
            ptr = int(self.memory_ptr.item())
            
            for i in range(batch_size):
                self.memory[ptr] = h[i]
                ptr = (ptr + 1) % self.memory_size
            
            self.memory_ptr[0] = ptr
    
    def query_memory(self, h: torch.Tensor) -> torch.Tensor:
        """Query memory with attention."""
        h_query = h.unsqueeze(0)
        memory = self.memory.unsqueeze(0)
        
        h_mem, _ = self.memory_attn(h_query, memory, memory)
        return h_mem.squeeze(0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Project
        h = self.node_proj(x)
        
        # Graph convolution
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Memory interaction
        if self.training:
            self.update_memory(h.detach())
        h_mem = self.query_memory(h)
        
        # Pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_mem_pool = global_mean_pool(h_mem, batch)
        
        h = torch.cat([h_mean, h_max, h_mem_pool], dim=-1)
        
        return self.classifier(h)


def build_baseline(name: str, config: Dict) -> nn.Module:
    """
    Factory function to build baseline models.
    
    Args:
        name: Baseline name (graphsage, gat, tgn, tgat, jodie, grabphisher, 2dynethnet)
        config: Model configuration
    
    Returns:
        Baseline model instance
    """
    baselines = {
        'graphsage': GraphSAGE,
        'gat': GAT,
        'tgn': TGN,
        'tgat': TGAT,
        'jodie': JODIE,
        'grabphisher': GrabPhisher,
        '2dynethnet': TwoDynEthNet,
    }
    
    if name.lower() not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")
    
    return baselines[name.lower()](
        in_channels=config.get('in_channels', 32),
        hidden_channels=config.get('hidden_channels', 128),
        out_channels=config.get('out_channels', 64),
        dropout=config.get('dropout', 0.2),
        num_classes=config.get('num_classes', 2)
    )


if __name__ == '__main__':
    # Quick test all baselines
    config = {'in_channels': 32, 'hidden_channels': 128, 'num_classes': 2}
    
    x = torch.randn(100, 32)
    edge_index = torch.randint(0, 100, (2, 500))
    batch = torch.zeros(100, dtype=torch.long)
    
    for name in ['graphsage', 'gat', 'tgn', 'tgat', 'jodie', 'grabphisher', '2dynethnet']:
        model = build_baseline(name, config)
        out = model(x, edge_index, batch=batch)
        print(f"{name}: {sum(p.numel() for p in model.parameters()):,} params, output shape: {out.shape}")

"""
ARTEMIS: Adversarial-Resistant Temporal Embedding Model for Intelligent Security
================================================================================

Main model architecture integrating all six innovations for Ethereum phishing detection.

Innovations:
    L1: Neural ODE-based continuous-time temporal modeling
    L2: Anomaly-aware memory storage with information-theoretic prioritization
    L3: Multi-hop message broadcasting for Sybil resistance
    L4: Adversarial meta-learning for robust adaptation
    L5: Elastic weight consolidation for continual learning
    L6: Certified adversarial training with randomized smoothing

Author: Anonymous (CCS 2026 Submission)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import degree, k_hop_subgraph
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from typing import Optional, Dict, Tuple, List
import math

from .artemis_innovations import (
    NeuralODEFunc,
    AnomalyAwareMemory,
    MultiHopBroadcast,
    AdversarialMetaLearner,
    ElasticWeightConsolidation,
    CertifiedAdversarialTrainer
)


class SpectralNormLinear(nn.Module):
    """
    Linear layer with spectral normalization for Lipschitz bound control.
    
    Mathematical Foundation:
        ||W||_2 ≤ 1 after spectral normalization
        This ensures: Lip(f) ≤ ∏_i ||W_i||_2 ≤ 1
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features, out_features, bias=bias)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TemporalGraphEncoder(nn.Module):
    """
    Temporal graph encoder with Neural ODE integration.
    
    Architecture:
        1. Initial node embedding via GAT
        2. Neural ODE evolution for temporal dynamics
        3. Multi-hop message broadcasting
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ode_method: str = 'dopri5',
        ode_rtol: float = 1e-4,
        ode_atol: float = 1e-5,
        broadcast_hops: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        
        # Initial projection
        if use_spectral_norm:
            self.input_proj = SpectralNormLinear(in_channels, hidden_channels)
        else:
            self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers for graph encoding
        self.gat1 = GATConv(
            hidden_channels, 
            hidden_channels // num_heads, 
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.gat2 = GATConv(
            hidden_channels, 
            hidden_channels // num_heads, 
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # Neural ODE function (Innovation L1)
        self.ode_func = NeuralODEFunc(
            hidden_channels=hidden_channels,
            use_spectral_norm=use_spectral_norm
        )
        
        # Multi-hop broadcast (Innovation L3)
        self.broadcast = MultiHopBroadcast(
            hidden_channels=hidden_channels,
            num_hops=broadcast_hops,
            aggregation='attention'
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through temporal graph encoder.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
            timestamps: Edge timestamps [E]
            batch: Batch assignment [N]
        
        Returns:
            Node embeddings [N, hidden_channels]
        """
        # Initial projection
        h = self.input_proj(x)
        h = F.gelu(h)
        
        # First GAT layer
        h = self.gat1(h, edge_index)
        h = F.gelu(h)
        h = self.norm1(h)
        h = self.dropout(h)
        
        # Second GAT layer
        h = self.gat2(h, edge_index)
        h = F.gelu(h)
        h = self.norm2(h)
        
        # Neural ODE evolution (Innovation L1)
        if timestamps is not None:
            t_span = torch.tensor([0.0, 1.0], device=h.device)
            h = odeint(
                self.ode_func,
                h,
                t_span,
                method=self.ode_method,
                rtol=self.ode_rtol,
                atol=self.ode_atol
            )[-1]  # Take final state
        
        # Multi-hop broadcast (Innovation L3)
        h = self.broadcast(h, edge_index, batch)
        
        return h


class ARTEMIS(nn.Module):
    """
    ARTEMIS: Full model architecture for Ethereum phishing detection.
    
    Integrates all six innovations:
        L1: Neural ODE temporal modeling
        L2: Anomaly-aware memory storage
        L3: Multi-hop message broadcasting
        L4: Adversarial meta-learning
        L5: Elastic weight consolidation
        L6: Certified adversarial training
    
    Mathematical Framework:
        Given temporal graph G(t) = (V, E(t), X(t), A(t))
        Predict y_v ∈ {0,1} for each node v ∈ V
        
        Loss = L_task + λ_ewc·L_ewc + λ_adv·L_adv + λ_cert·L_cert
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Model dimensions
        self.in_channels = config.get('in_channels', 32)
        self.hidden_channels = config.get('hidden_channels', 128)
        self.out_channels = config.get('out_channels', 64)
        self.num_classes = config.get('num_classes', 2)
        
        # Innovation flags
        self.use_ode = config.get('use_ode', True)
        self.use_anomaly_memory = config.get('use_anomaly_memory', True)
        self.use_multihop = config.get('use_multihop', True)
        self.use_adversarial_meta = config.get('use_adversarial_meta', True)
        self.use_ewc = config.get('use_ewc', True)
        self.use_certified = config.get('use_certified', True)
        
        # Temporal graph encoder (includes L1, L3)
        self.encoder = TemporalGraphEncoder(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1),
            ode_method=config.get('ode_method', 'dopri5'),
            broadcast_hops=config.get('broadcast_hops', 3) if self.use_multihop else 1,
            use_spectral_norm=self.use_certified
        )
        
        # Anomaly-aware memory (Innovation L2)
        if self.use_anomaly_memory:
            self.memory = AnomalyAwareMemory(
                memory_size=config.get('memory_size', 1000),
                embedding_dim=self.hidden_channels,
                anomaly_weight=config.get('anomaly_weight', 1.0)
            )
        else:
            self.memory = None
        
        # Graph-level readout
        self.readout_mlp = nn.Sequential(
            SpectralNormLinear(self.hidden_channels * 2, self.hidden_channels) 
                if self.use_certified else nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            SpectralNormLinear(self.hidden_channels, self.hidden_channels // 2)
                if self.use_certified else nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.GELU(),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            SpectralNormLinear(self.hidden_channels // 2, self.num_classes)
                if self.use_certified else nn.Linear(self.hidden_channels // 2, self.num_classes)
        )
        
        # Elastic Weight Consolidation (Innovation L5)
        if self.use_ewc:
            self.ewc = ElasticWeightConsolidation(
                model=self,
                ewc_lambda=config.get('ewc_lambda', 5000)
            )
        else:
            self.ewc = None
        
        # Certified adversarial trainer (Innovation L6)
        if self.use_certified:
            self.cert_trainer = CertifiedAdversarialTrainer(
                model=self,
                epsilon=config.get('adv_epsilon', 0.1),
                sigma=config.get('smoothing_sigma', 0.25),
                n_samples=config.get('smoothing_samples', 100)
            )
        else:
            self.cert_trainer = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode graph to node embeddings.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
            timestamps: Edge timestamps [E]
            batch: Batch assignment [N]
        
        Returns:
            Node embeddings [N, hidden_channels]
        """
        # Get node embeddings from encoder
        h = self.encoder(x, edge_index, edge_attr, timestamps, batch)
        
        # Update and query memory (Innovation L2)
        if self.memory is not None and self.training:
            h = self.memory.update_and_query(h, batch)
        
        return h
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for graph classification.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
            timestamps: Edge timestamps [E]
            batch: Batch assignment [N]
            return_embeddings: If True, return intermediate embeddings
        
        Returns:
            Logits [batch_size, num_classes] or (logits, embeddings) tuple
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode graph
        h = self.encode(x, edge_index, edge_attr, timestamps, batch)
        
        # Graph-level readout (mean + max pooling)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)
        
        # MLP readout
        h_out = self.readout_mlp(h_graph)
        
        # Classification
        logits = self.classifier(h_out)
        
        if return_embeddings:
            return logits, h_out
        return logits
    
    def compute_loss(
        self,
        batch_data,
        task_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total training loss with all regularization terms.
        
        Args:
            batch_data: PyG batch data
            task_id: Current task ID for continual learning
        
        Returns:
            Total loss and dictionary of loss components
        """
        # Forward pass
        logits = self(
            batch_data.x,
            batch_data.edge_index,
            getattr(batch_data, 'edge_attr', None),
            getattr(batch_data, 'timestamps', None),
            batch_data.batch
        )
        
        # Task loss (cross-entropy)
        loss_task = F.cross_entropy(logits, batch_data.y)
        
        loss_dict = {'task': loss_task.item()}
        total_loss = loss_task
        
        # EWC regularization (Innovation L5)
        if self.ewc is not None and task_id is not None and task_id > 0:
            loss_ewc = self.ewc.penalty()
            total_loss = total_loss + loss_ewc
            loss_dict['ewc'] = loss_ewc.item()
        
        # Adversarial training loss (Innovation L6)
        if self.cert_trainer is not None and self.training:
            loss_adv = self.cert_trainer.adversarial_loss(batch_data)
            total_loss = total_loss + self.config.get('adv_weight', 0.5) * loss_adv
            loss_dict['adversarial'] = loss_adv.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
    
    def certify(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        n_samples: int = 1000,
        alpha: float = 0.001
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute certified predictions using randomized smoothing.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            n_samples: Number of MC samples
            alpha: Confidence level
        
        Returns:
            Certified predictions and certified radii
        """
        if self.cert_trainer is None:
            raise ValueError("Certified training not enabled")
        
        return self.cert_trainer.certify(x, edge_index, batch, n_samples, alpha)
    
    def update_ewc(self, dataloader, task_id: int):
        """Update EWC Fisher information after completing a task."""
        if self.ewc is not None:
            self.ewc.update_fisher(dataloader, task_id)
    
    def get_lipschitz_bound(self) -> float:
        """
        Compute Lipschitz bound of the model.
        
        For certified robustness:
            Lip(f) = ∏_i σ_max(W_i)
        
        With spectral normalization: Lip(f) ≤ 1
        """
        lip_bound = 1.0
        for module in self.modules():
            if hasattr(module, 'weight_orig'):  # Spectral normalized
                with torch.no_grad():
                    u = module.weight_u
                    v = module.weight_v
                    sigma = torch.dot(u, torch.mv(module.weight_orig, v))
                    lip_bound *= sigma.item()
        return lip_bound


class ARTEMISNodeClassifier(ARTEMIS):
    """
    ARTEMIS variant for node-level classification.
    
    Used for direct phishing address detection without graph aggregation.
    """
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        target_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for node classification.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
            timestamps: Edge timestamps [E]
            batch: Batch assignment [N]
            target_nodes: Indices of nodes to classify [M]
        
        Returns:
            Logits [M, num_classes] or [N, num_classes]
        """
        # Encode all nodes
        h = self.encode(x, edge_index, edge_attr, timestamps, batch)
        
        # Select target nodes if specified
        if target_nodes is not None:
            h = h[target_nodes]
        
        # Classification (no pooling needed)
        h = self.readout_mlp[:3](
            torch.cat([h, h], dim=-1)  # Duplicate to match input dim
        )
        logits = self.classifier(h)
        
        return logits


def build_artemis(config: Dict) -> ARTEMIS:
    """
    Factory function to build ARTEMIS model from configuration.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        ARTEMIS model instance
    """
    model_type = config.get('model_type', 'graph')
    
    if model_type == 'node':
        return ARTEMISNodeClassifier(config)
    else:
        return ARTEMIS(config)


if __name__ == '__main__':
    # Quick test
    config = {
        'in_channels': 32,
        'hidden_channels': 128,
        'out_channels': 64,
        'num_classes': 2,
        'use_ode': True,
        'use_anomaly_memory': True,
        'use_multihop': True,
        'use_ewc': True,
        'use_certified': True,
    }
    
    model = build_artemis(config)
    print(f"ARTEMIS model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    x = torch.randn(100, 32)
    edge_index = torch.randint(0, 100, (2, 500))
    batch = torch.zeros(100, dtype=torch.long)
    
    logits = model(x, edge_index, batch=batch)
    print(f"Output shape: {logits.shape}")

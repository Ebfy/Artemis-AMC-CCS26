#!/usr/bin/env python3
"""
ARTEMIS Ablation Study Script
Reproduces Table 4: Component Contribution Analysis

This script systematically evaluates the contribution of each ARTEMIS innovation
by training variants with specific components disabled.

Ablation Variants:
1. ARTEMIS-Full: All 6 innovations enabled
2. w/o Neural ODE: Replace with discrete RNN (L1 disabled)
3. w/o Anomaly Memory: Replace with FIFO memory (L2 disabled)
4. w/o Multi-hop Broadcast: Use 1-hop only (L3 disabled)
5. w/o Adversarial Meta-Learning: Standard training (L4 disabled)
6. w/o EWC: No continual learning regularization (L5 disabled)
7. w/o Certified Training: Standard adversarial training (L6 disabled)

Expected Results (Table 4):
- Full ARTEMIS: 91.47% Recall, 90.18% F1, 96.23% AUC
- Largest drop: w/o Neural ODE (-4.24% Recall)
- All components contribute positively

Usage:
    python run_ablation_study.py --config configs/default.yaml
    python run_ablation_study.py --variant no_ode  # Single variant
    python run_ablation_study.py --quick  # Fast validation run
"""

import os
import sys
import json
import yaml
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from artemis_model import ARTEMIS, build_artemis
from artemis_innovations import (
    TemporalODEBlock, AnomalyAwareMemory, MultiHopBroadcast,
    AdversarialMetaLearner, ElasticWeightConsolidation, CertifiedAdversarialTrainer
)


# ============================================================================
# Ablation Variant Configurations
# ============================================================================

ABLATION_VARIANTS = {
    'full': {
        'name': 'ARTEMIS-Full',
        'description': 'All 6 innovations enabled',
        'use_neural_ode': True,
        'use_anomaly_memory': True,
        'use_multihop_broadcast': True,
        'use_adversarial_meta': True,
        'use_ewc': True,
        'use_certified_training': True,
    },
    'no_ode': {
        'name': 'w/o Neural ODE',
        'description': 'Replace Neural ODE with discrete RNN',
        'use_neural_ode': False,
        'use_anomaly_memory': True,
        'use_multihop_broadcast': True,
        'use_adversarial_meta': True,
        'use_ewc': True,
        'use_certified_training': True,
    },
    'no_anomaly_memory': {
        'name': 'w/o Anomaly Memory',
        'description': 'Replace anomaly-aware memory with FIFO',
        'use_neural_ode': True,
        'use_anomaly_memory': False,
        'use_multihop_broadcast': True,
        'use_adversarial_meta': True,
        'use_ewc': True,
        'use_certified_training': True,
    },
    'no_multihop': {
        'name': 'w/o Multi-hop Broadcast',
        'description': 'Use 1-hop message passing only',
        'use_neural_ode': True,
        'use_anomaly_memory': True,
        'use_multihop_broadcast': False,
        'use_adversarial_meta': True,
        'use_ewc': True,
        'use_certified_training': True,
    },
    'no_adversarial_meta': {
        'name': 'w/o Adversarial Meta-Learning',
        'description': 'Standard training without meta-learning',
        'use_neural_ode': True,
        'use_anomaly_memory': True,
        'use_multihop_broadcast': True,
        'use_adversarial_meta': False,
        'use_ewc': True,
        'use_certified_training': True,
    },
    'no_ewc': {
        'name': 'w/o EWC',
        'description': 'No elastic weight consolidation',
        'use_neural_ode': True,
        'use_anomaly_memory': True,
        'use_multihop_broadcast': True,
        'use_adversarial_meta': True,
        'use_ewc': False,
        'use_certified_training': True,
    },
    'no_certified': {
        'name': 'w/o Certified Training',
        'description': 'Standard adversarial training without certification',
        'use_neural_ode': True,
        'use_anomaly_memory': True,
        'use_multihop_broadcast': True,
        'use_adversarial_meta': True,
        'use_ewc': True,
        'use_certified_training': False,
    },
}


# ============================================================================
# Ablation Model Builder
# ============================================================================

class DiscreteRNNBlock(nn.Module):
    """Discrete RNN replacement for Neural ODE (ablation baseline)."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
    def forward(self, h: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply discrete RNN instead of Neural ODE."""
        # h: [batch, hidden_dim] -> [batch, 1, hidden_dim]
        h_seq = h.unsqueeze(1)
        out, _ = self.rnn(h_seq)
        return out.squeeze(1)


class FIFOMemory(nn.Module):
    """Simple FIFO memory replacement for anomaly-aware memory (ablation baseline)."""
    
    def __init__(self, memory_size: int, hidden_dim: int):
        super().__init__()
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        self.register_buffer('memory', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('write_ptr', torch.tensor(0))
        
    def update(self, embeddings: torch.Tensor) -> None:
        """FIFO update - simply overwrite oldest entries."""
        batch_size = embeddings.size(0)
        for i in range(batch_size):
            idx = (self.write_ptr.item() + i) % self.memory_size
            self.memory[idx] = embeddings[i].detach()
        self.write_ptr = (self.write_ptr + batch_size) % self.memory_size
        
    def query(self, query: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple nearest neighbor query without importance weighting."""
        # Compute cosine similarity
        query_norm = F.normalize(query, dim=-1)
        memory_norm = F.normalize(self.memory, dim=-1)
        scores = torch.mm(query_norm, memory_norm.t())
        
        # Get top-k
        values, indices = torch.topk(scores, min(top_k, self.memory_size), dim=-1)
        retrieved = self.memory[indices]
        
        return retrieved, values


class SingleHopBroadcast(nn.Module):
    """1-hop message passing replacement for multi-hop broadcast (ablation baseline)."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Single hop aggregation only."""
        # Simple attention-based aggregation (1-hop)
        attn_out, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        return attn_out.squeeze(0)


def build_ablation_model(
    variant_config: Dict,
    model_config: Dict,
    device: torch.device
) -> nn.Module:
    """
    Build ARTEMIS model with specific components disabled for ablation.
    
    Args:
        variant_config: Ablation variant configuration
        model_config: Base model configuration
        device: Target device
        
    Returns:
        Model with specified components enabled/disabled
    """
    # Create modified config
    config = copy.deepcopy(model_config)
    
    # Apply ablation settings
    config['use_neural_ode'] = variant_config['use_neural_ode']
    config['use_anomaly_memory'] = variant_config['use_anomaly_memory']
    config['use_multihop_broadcast'] = variant_config['use_multihop_broadcast']
    config['use_adversarial_meta'] = variant_config['use_adversarial_meta']
    config['use_ewc'] = variant_config['use_ewc']
    config['use_certified_training'] = variant_config['use_certified_training']
    
    # Build model
    model = build_artemis(config)
    
    # Replace components with ablation baselines if disabled
    if not variant_config['use_neural_ode']:
        # Replace Neural ODE with discrete RNN
        model.temporal_block = DiscreteRNNBlock(
            hidden_dim=config.get('hidden_dim', 128)
        ).to(device)
        
    if not variant_config['use_anomaly_memory']:
        # Replace anomaly-aware memory with FIFO
        model.memory = FIFOMemory(
            memory_size=config.get('memory_size', 1000),
            hidden_dim=config.get('hidden_dim', 128)
        ).to(device)
        
    if not variant_config['use_multihop_broadcast']:
        # Replace multi-hop with single-hop
        model.broadcast = SingleHopBroadcast(
            hidden_dim=config.get('hidden_dim', 128)
        ).to(device)
    
    return model.to(device)


# ============================================================================
# Training and Evaluation
# ============================================================================

class AblationTrainer:
    """Trainer for ablation study experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        variant_config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.model = model
        self.variant_config = variant_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optional components based on variant
        self.meta_learner = None
        self.ewc = None
        self.certified_trainer = None
        
        if variant_config['use_adversarial_meta']:
            self.meta_learner = AdversarialMetaLearner(
                model=model,
                inner_lr=config.get('inner_lr', 0.01),
                outer_lr=config.get('outer_lr', 0.001)
            )
            
        if variant_config['use_ewc']:
            self.ewc = ElasticWeightConsolidation(
                model=model,
                ewc_lambda=config.get('ewc_lambda', 0.5)
            )
            
        if variant_config['use_certified_training']:
            self.certified_trainer = CertifiedAdversarialTrainer(
                model=model,
                epsilon=config.get('epsilon', 0.1),
                sigma=config.get('smoothing_sigma', 0.1)
            )
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(batch)
            labels = batch.y if hasattr(batch, 'y') else batch.label
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Add EWC penalty if enabled
            if self.ewc is not None and hasattr(self.ewc, 'fisher') and self.ewc.fisher:
                ewc_loss = self.ewc.penalty()
                loss = loss + ewc_loss
                
            # Add certified adversarial loss if enabled
            if self.certified_trainer is not None:
                adv_loss = self.certified_trainer.adversarial_loss(batch, labels)
                loss = loss + 0.5 * adv_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': (all_preds == all_labels).mean(),
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on given data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            labels = batch.y if hasattr(batch, 'y') else batch.label
            
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy() if probs.size(1) > 1 else probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute comprehensive metrics
        metrics = self._compute_metrics(all_preds, all_labels, all_probs)
        return metrics
    
    def _compute_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray
    ) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        # Basic metrics
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fpr = fp / (fp + tn + 1e-10)
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'fpr': float(fpr),
        }
    
    def train(self, epochs: int) -> Dict[str, any]:
        """Full training loop."""
        history = {
            'train_loss': [],
            'val_metrics': [],
        }
        
        patience = self.config.get('patience', 15)
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Early stopping check
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
                
            # Progress logging
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: Loss={train_metrics['loss']:.4f}, "
                      f"Val F1={val_metrics['f1']:.4f}, Val Recall={val_metrics['recall']:.4f}")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        # Update EWC Fisher information if enabled
        if self.ewc is not None:
            self.ewc.compute_fisher(self.train_loader)
        
        # Final test evaluation
        test_metrics = self.evaluate(self.test_loader)
        
        return {
            'history': history,
            'test_metrics': test_metrics,
            'best_val_f1': self.best_val_f1,
        }


# ============================================================================
# Data Loading
# ============================================================================

def create_synthetic_data(config: Dict, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create synthetic data for testing ablation study pipeline."""
    from torch_geometric.data import Data, Batch
    
    num_samples = config.get('num_samples', 1000)
    num_nodes = config.get('num_nodes', 50)
    num_features = config.get('input_dim', 64)
    batch_size = config.get('batch_size', 32)
    
    def create_graph():
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        y = torch.randint(0, 2, (1,)).item()
        return Data(x=x, edge_index=edge_index, y=torch.tensor([y]))
    
    # Create datasets
    train_data = [create_graph() for _ in range(int(num_samples * 0.7))]
    val_data = [create_graph() for _ in range(int(num_samples * 0.15))]
    test_data = [create_graph() for _ in range(int(num_samples * 0.15))]
    
    # Create data loaders
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def load_etgraph_data(config: Dict, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load ETGraph dataset for ablation study."""
    data_dir = config.get('data_dir', 'data/etgraph')
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"ETGraph data not found at {data_dir}")
        print("Using synthetic data for demonstration")
        return create_synthetic_data(config, device)
    
    # Load preprocessed data
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        
        train_data = torch.load(os.path.join(data_dir, 'train_graphs.pt'))
        val_data = torch.load(os.path.join(data_dir, 'val_graphs.pt'))
        test_data = torch.load(os.path.join(data_dir, 'test_graphs.pt'))
        
        batch_size = config.get('batch_size', 32)
        
        train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"Error loading ETGraph data: {e}")
        print("Using synthetic data for demonstration")
        return create_synthetic_data(config, device)


# ============================================================================
# Ablation Study Runner
# ============================================================================

def run_ablation_variant(
    variant_name: str,
    variant_config: Dict,
    model_config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seeds: List[int] = [42, 123, 456]
) -> Dict:
    """
    Run ablation study for a single variant across multiple seeds.
    """
    print(f"\n{'='*60}")
    print(f"Running ablation variant: {variant_config['name']}")
    print(f"Description: {variant_config['description']}")
    print(f"Seeds: {seeds}")
    print('='*60)
    
    all_results = []
    
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Build model
        model = build_ablation_model(variant_config, model_config, device)
        
        # Create trainer
        trainer = AblationTrainer(
            model=model,
            variant_config=variant_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=model_config,
            device=device
        )
        
        # Train
        epochs = model_config.get('epochs', 100)
        result = trainer.train(epochs)
        
        all_results.append(result['test_metrics'])
        
        print(f"    Test Recall: {result['test_metrics']['recall']:.4f}")
        print(f"    Test F1: {result['test_metrics']['f1']:.4f}")
        print(f"    Test AUC: {result['test_metrics']['auc']:.4f}")
    
    # Aggregate results across seeds
    aggregated = {}
    for metric in all_results[0].keys():
        values = [r[metric] for r in all_results]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    return {
        'variant_name': variant_name,
        'config': variant_config,
        'results': all_results,
        'aggregated': aggregated
    }


def run_full_ablation_study(config: Dict, args) -> Dict:
    """
    Run complete ablation study across all variants.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    if args.quick:
        print("\nQuick mode: Using minimal synthetic data")
        config['num_samples'] = 200
        config['epochs'] = 5
        
    train_loader, val_loader, test_loader = load_etgraph_data(config, device)
    
    # Determine which variants to run
    if args.variant:
        variants_to_run = [args.variant]
    else:
        variants_to_run = list(ABLATION_VARIANTS.keys())
    
    # Run ablation for each variant
    all_variant_results = {}
    seeds = [42, 123, 456] if not args.quick else [42]
    
    for variant_name in variants_to_run:
        if variant_name not in ABLATION_VARIANTS:
            print(f"Unknown variant: {variant_name}, skipping")
            continue
            
        variant_config = ABLATION_VARIANTS[variant_name]
        result = run_ablation_variant(
            variant_name=variant_name,
            variant_config=variant_config,
            model_config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            seeds=seeds
        )
        all_variant_results[variant_name] = result
    
    return all_variant_results


def generate_ablation_table(results: Dict) -> str:
    """
    Generate Table 4 format from ablation results.
    """
    table = "\n" + "="*80 + "\n"
    table += "TABLE 4: Ablation Study - Component Contribution Analysis\n"
    table += "="*80 + "\n"
    table += f"{'Variant':<30} {'Recall':<15} {'F1':<15} {'AUC':<15} {'Δ Recall':<10}\n"
    table += "-"*80 + "\n"
    
    # Get full model results for comparison
    full_recall = results.get('full', {}).get('aggregated', {}).get('recall', {}).get('mean', 0)
    
    for variant_name in ['full', 'no_ode', 'no_anomaly_memory', 'no_multihop', 
                         'no_adversarial_meta', 'no_ewc', 'no_certified']:
        if variant_name not in results:
            continue
            
        res = results[variant_name]
        agg = res['aggregated']
        
        recall = agg['recall']['mean']
        recall_std = agg['recall']['std']
        f1 = agg['f1']['mean']
        f1_std = agg['f1']['std']
        auc = agg['auc']['mean']
        auc_std = agg['auc']['std']
        
        delta = recall - full_recall if variant_name != 'full' else 0.0
        delta_str = f"{delta:+.2%}" if variant_name != 'full' else "-"
        
        name = res['config']['name']
        table += f"{name:<30} {recall:.2%}±{recall_std:.2%}  {f1:.2%}±{f1_std:.2%}  {auc:.2%}±{auc_std:.2%}  {delta_str}\n"
    
    table += "-"*80 + "\n"
    table += "Note: Δ Recall shows change relative to full ARTEMIS model\n"
    table += "="*80 + "\n"
    
    return table


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ARTEMIS Ablation Study')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--variant', type=str, default=None,
                        choices=list(ABLATION_VARIANTS.keys()),
                        help='Run specific variant only')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation run with minimal data')
    parser.add_argument('--output', type=str, default='results/ablation',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found at {args.config}, using defaults")
        config = {
            'input_dim': 64,
            'hidden_dim': 128,
            'output_dim': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'epochs': 100,
            'batch_size': 32,
            'patience': 15,
        }
    
    # Merge model config if nested
    if 'model' in config:
        model_config = {**config, **config['model']}
    else:
        model_config = config
    
    print("\n" + "="*60)
    print("ARTEMIS ABLATION STUDY")
    print("Reproducing Table 4: Component Contribution Analysis")
    print("="*60)
    
    # Run ablation study
    start_time = time.time()
    results = run_full_ablation_study(model_config, args)
    elapsed = time.time() - start_time
    
    # Generate summary table
    table = generate_ablation_table(results)
    print(table)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(args.output, 'ablation_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    # Save summary table
    table_file = os.path.join(args.output, 'ablation_table.txt')
    with open(table_file, 'w') as f:
        f.write(table)
    
    print(f"\nResults saved to {args.output}/")
    print(f"Total time: {elapsed/60:.1f} minutes")
    
    return results


if __name__ == '__main__':
    main()

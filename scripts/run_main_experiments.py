#!/usr/bin/env python3
"""
ARTEMIS Main Experiments Script
================================

Runs the main evaluation experiments for Table 3 in the paper.

Usage:
    python scripts/run_main_experiments.py --config configs/default.yaml
    python scripts/run_main_experiments.py --config configs/default.yaml --quick
    python scripts/run_main_experiments.py --config configs/default.yaml --task 1

Author: Anonymous (CCS 2026 Submission)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import yaml
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from artemis_model import build_artemis
from baseline_implementations import build_baseline


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
    
    Returns:
        Dictionary of metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # TPR
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return metrics


def statistical_significance(
    artemis_scores: list,
    baseline_scores: list,
    alpha: float = 0.01
) -> dict:
    """
    Compute statistical significance tests.
    
    Args:
        artemis_scores: ARTEMIS scores across runs
        baseline_scores: Baseline scores across runs
        alpha: Significance level
    
    Returns:
        Dictionary with p-value, t-statistic, Cohen's d
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(artemis_scores, baseline_scores)
    
    # Wilcoxon signed-rank test
    w_stat, w_pvalue = stats.wilcoxon(artemis_scores, baseline_scores)
    
    # Cohen's d
    diff = np.array(artemis_scores) - np.array(baseline_scores)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pvalue': float(w_pvalue),
        'cohens_d': float(cohens_d),
        'significant': p_value < alpha
    }


class Trainer:
    """Training and evaluation manager for ARTEMIS."""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and config['hardware'].get('use_multi_gpu', False):
            self.model = nn.DataParallel(self.model)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['min_lr']
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        
        # Metrics storage
        self.train_losses = []
        self.val_metrics = []
    
    def train_epoch(self, dataloader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        batch.x, batch.edge_index,
                        edge_attr=getattr(batch, 'edge_attr', None),
                        timestamps=getattr(batch, 'timestamps', None),
                        batch=batch.batch
                    )
                    loss = F.cross_entropy(logits, batch.y)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['clip_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(
                    batch.x, batch.edge_index,
                    edge_attr=getattr(batch, 'edge_attr', None),
                    timestamps=getattr(batch, 'timestamps', None),
                    batch=batch.batch
                )
                loss = F.cross_entropy(logits, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['clip_grad_norm']
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        self.scheduler.step()
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """Evaluate on dataset."""
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            logits = self.model(
                batch.x, batch.edge_index,
                edge_attr=getattr(batch, 'edge_attr', None),
                timestamps=getattr(batch, 'timestamps', None),
                batch=batch.batch
            )
            
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return metrics
    
    def fit(self, train_loader, val_loader, num_epochs: int) -> dict:
        """Full training loop with early stopping."""
        best_val_f1 = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            self.val_metrics.append(val_metrics)
            
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, "
                  f"Val Recall={val_metrics['recall']:.4f}, "
                  f"Val F1={val_metrics['f1']:.4f}, "
                  f"Val AUC={val_metrics['auc']:.4f}")
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1 + self.config['training']['min_delta']:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['training']['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {'best_val_f1': best_val_f1, 'epochs_trained': epoch}


def create_synthetic_data(num_graphs: int, num_nodes: int = 100, num_features: int = 32):
    """
    Create synthetic data for testing (when ETGraph not available).
    
    In actual experiments, this is replaced with ETGraph dataset.
    """
    from torch_geometric.data import Data, Batch
    
    data_list = []
    for i in range(num_graphs):
        # Random graph
        num_edges = num_nodes * 5
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, 2, (1,))  # Graph label
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list


def main():
    parser = argparse.ArgumentParser(description="ARTEMIS Main Experiments")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with reduced data')
    parser.add_argument('--task', type=int, default=None,
                       help='Run specific task only (1-6)')
    parser.add_argument('--output', type=str, default='results/main_results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine tasks to run
    if args.task is not None:
        task_ids = [args.task]
    else:
        task_ids = list(range(1, config['data']['num_tasks'] + 1))
    
    # Quick mode adjustments
    if args.quick:
        config['training']['epochs'] = 5
        num_graphs = 100
    else:
        num_graphs = 1000
    
    # Results storage
    all_results = {
        'artemis': {},
        'baselines': {},
        'comparison': {},
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    # Run experiments for each task
    for task_id in task_ids:
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {config['data']['tasks'][f'task_{task_id}']['description']}")
        print(f"{'='*60}")
        
        # Create/load data (using synthetic for demo)
        print("Loading data...")
        train_data = create_synthetic_data(int(num_graphs * 0.7), num_features=config['model']['in_channels'])
        val_data = create_synthetic_data(int(num_graphs * 0.15), num_features=config['model']['in_channels'])
        test_data = create_synthetic_data(int(num_graphs * 0.15), num_features=config['model']['in_channels'])
        
        train_loader = PyGDataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = PyGDataLoader(val_data, batch_size=config['training']['batch_size'])
        test_loader = PyGDataLoader(test_data, batch_size=config['training']['batch_size'])
        
        # Train ARTEMIS
        print("\nTraining ARTEMIS...")
        artemis_model = build_artemis(config['model'])
        trainer = Trainer(artemis_model, config, device)
        
        start_time = time.time()
        train_info = trainer.fit(train_loader, val_loader, config['training']['epochs'])
        training_time = time.time() - start_time
        
        # Evaluate ARTEMIS
        test_metrics = trainer.evaluate(test_loader)
        test_metrics['training_time'] = training_time
        test_metrics['num_parameters'] = sum(p.numel() for p in artemis_model.parameters())
        
        all_results['artemis'][f'task_{task_id}'] = test_metrics
        print(f"\nARTEMIS Results:")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  FPR: {test_metrics['fpr']:.4f}")
        
        # Train and evaluate baselines
        all_results['baselines'][f'task_{task_id}'] = {}
        
        for baseline_name in ['graphsage', 'gat', 'tgn', '2dynethnet']:
            print(f"\nTraining {baseline_name}...")
            
            baseline_model = build_baseline(baseline_name, config['model'])
            baseline_trainer = Trainer(baseline_model, config, device)
            
            start_time = time.time()
            baseline_trainer.fit(train_loader, val_loader, config['training']['epochs'])
            baseline_training_time = time.time() - start_time
            
            baseline_metrics = baseline_trainer.evaluate(test_loader)
            baseline_metrics['training_time'] = baseline_training_time
            baseline_metrics['num_parameters'] = sum(p.numel() for p in baseline_model.parameters())
            
            all_results['baselines'][f'task_{task_id}'][baseline_name] = baseline_metrics
            
            print(f"  {baseline_name} - Recall: {baseline_metrics['recall']:.4f}, "
                  f"F1: {baseline_metrics['f1']:.4f}, AUC: {baseline_metrics['auc']:.4f}")
    
    # Compute aggregated results
    print("\n" + "="*60)
    print("AGGREGATED RESULTS ACROSS ALL TASKS")
    print("="*60)
    
    # ARTEMIS aggregation
    artemis_recalls = [all_results['artemis'][f'task_{t}']['recall'] for t in task_ids]
    artemis_f1s = [all_results['artemis'][f'task_{t}']['f1'] for t in task_ids]
    artemis_aucs = [all_results['artemis'][f'task_{t}']['auc'] for t in task_ids]
    
    all_results['artemis']['aggregated'] = {
        'recall_mean': float(np.mean(artemis_recalls)),
        'recall_std': float(np.std(artemis_recalls)),
        'f1_mean': float(np.mean(artemis_f1s)),
        'f1_std': float(np.std(artemis_f1s)),
        'auc_mean': float(np.mean(artemis_aucs)),
        'auc_std': float(np.std(artemis_aucs))
    }
    
    print(f"\nARTEMIS (aggregated):")
    print(f"  Recall: {np.mean(artemis_recalls):.4f} ± {np.std(artemis_recalls):.4f}")
    print(f"  F1: {np.mean(artemis_f1s):.4f} ± {np.std(artemis_f1s):.4f}")
    print(f"  AUC: {np.mean(artemis_aucs):.4f} ± {np.std(artemis_aucs):.4f}")
    
    # Baseline aggregation and comparison
    for baseline_name in ['graphsage', 'gat', 'tgn', '2dynethnet']:
        baseline_recalls = [all_results['baselines'][f'task_{t}'][baseline_name]['recall'] for t in task_ids]
        baseline_f1s = [all_results['baselines'][f'task_{t}'][baseline_name]['f1'] for t in task_ids]
        
        all_results['baselines']['aggregated'] = all_results['baselines'].get('aggregated', {})
        all_results['baselines']['aggregated'][baseline_name] = {
            'recall_mean': float(np.mean(baseline_recalls)),
            'recall_std': float(np.std(baseline_recalls)),
            'f1_mean': float(np.mean(baseline_f1s)),
            'f1_std': float(np.std(baseline_f1s))
        }
        
        # Statistical significance vs ARTEMIS
        sig_results = statistical_significance(artemis_recalls, baseline_recalls)
        all_results['comparison'][f'artemis_vs_{baseline_name}'] = sig_results
        
        improvement = (np.mean(artemis_recalls) - np.mean(baseline_recalls)) / np.mean(baseline_recalls) * 100
        
        print(f"\n{baseline_name}:")
        print(f"  Recall: {np.mean(baseline_recalls):.4f} ± {np.std(baseline_recalls):.4f}")
        print(f"  ARTEMIS improvement: +{improvement:.2f}%")
        print(f"  Statistical significance: p={sig_results['p_value']:.4f}, Cohen's d={sig_results['cohens_d']:.2f}")
    
    # Save results
    results_file = output_dir / 'metrics.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("ARTEMIS Main Experiment Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"ARTEMIS Recall: {np.mean(artemis_recalls):.4f} ± {np.std(artemis_recalls):.4f}\n")
        f.write(f"ARTEMIS F1: {np.mean(artemis_f1s):.4f} ± {np.std(artemis_f1s):.4f}\n")
        f.write(f"ARTEMIS AUC: {np.mean(artemis_aucs):.4f} ± {np.std(artemis_aucs):.4f}\n")
    print(f"Summary saved to {summary_file}")


if __name__ == '__main__':
    main()

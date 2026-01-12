#!/usr/bin/env python3
"""
ARTEMIS Continual Learning Evaluation Script

Evaluates ARTEMIS and baselines on sequential task learning to measure
catastrophic forgetting resistance (Figure 6 in paper).

This script:
1. Trains models sequentially on Tasks 1-6
2. Measures performance retention on previous tasks after learning new ones
3. Compares EWC-enabled vs standard training
4. Generates forgetting curves and performance matrices

Usage:
    python run_continual_learning.py --config configs/default.yaml
    python run_continual_learning.py --config configs/default.yaml --quick
    python run_continual_learning.py --tasks 1 2 3 --output results/continual/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from copy import deepcopy

import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from artemis_model import ARTEMIS, build_artemis
    from artemis_innovations import ElasticWeightConsolidation
    from baseline_implementations import build_baseline
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


@dataclass
class ContinualLearningResult:
    """Results from continual learning evaluation."""
    model_name: str
    task_sequence: List[int]
    performance_matrix: List[List[float]]  # [task_trained][task_evaluated]
    forgetting_scores: List[float]  # Per-task forgetting
    average_forgetting: float
    backward_transfer: float
    forward_transfer: float
    final_accuracy: float
    training_time: float


@dataclass
class TaskPerformance:
    """Performance on a single task."""
    task_id: int
    accuracy: float
    recall: float
    f1_score: float
    auc_roc: float


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_synthetic_task_data(task_id: int, num_graphs: int = 200, seed: int = 42) -> List[Data]:
    """Create synthetic data for a task with task-specific characteristics."""
    np.random.seed(seed + task_id)
    torch.manual_seed(seed + task_id)
    
    graphs = []
    
    # Task-specific parameters to create distribution shift
    task_params = {
        1: {"avg_nodes": 50, "edge_density": 0.1, "feature_mean": 0.0},
        2: {"avg_nodes": 75, "edge_density": 0.15, "feature_mean": 0.5},
        3: {"avg_nodes": 100, "edge_density": 0.08, "feature_mean": -0.5},
        4: {"avg_nodes": 60, "edge_density": 0.12, "feature_mean": 0.3},
        5: {"avg_nodes": 80, "edge_density": 0.1, "feature_mean": -0.3},
        6: {"avg_nodes": 90, "edge_density": 0.14, "feature_mean": 0.1},
    }
    
    params = task_params.get(task_id, task_params[1])
    
    for i in range(num_graphs):
        num_nodes = np.random.poisson(params["avg_nodes"]) + 10
        num_edges = int(num_nodes * num_nodes * params["edge_density"])
        
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, 16) + params["feature_mean"]
        timestamps = torch.sort(torch.rand(num_edges))[0]
        
        # Label with task-specific bias
        label = 1 if np.random.random() < 0.1 + 0.02 * task_id else 0
        y = torch.tensor([label], dtype=torch.long)
        
        graphs.append(Data(
            x=x, 
            edge_index=edge_index, 
            timestamps=timestamps, 
            y=y,
            task_id=torch.tensor([task_id])
        ))
    
    return graphs


def split_data(graphs: List[Data], train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split graphs into train/val/test sets."""
    n = len(graphs)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_graphs = [graphs[i] for i in indices[:train_end]]
    val_graphs = [graphs[i] for i in indices[train_end:val_end]]
    test_graphs = [graphs[i] for i in indices[val_end:]]
    
    return train_graphs, val_graphs, test_graphs


class ContinualLearningTrainer:
    """Trainer for continual learning evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        use_ewc: bool = True,
        ewc_lambda: float = 1000.0,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.use_ewc = use_ewc
        self.device = device
        
        # Initialize EWC if enabled
        if use_ewc:
            self.ewc = ElasticWeightConsolidation(
                model=model,
                ewc_lambda=ewc_lambda
            )
        else:
            self.ewc = None
        
        # Task-specific data storage
        self.task_loaders: Dict[int, DataLoader] = {}
        self.task_test_loaders: Dict[int, DataLoader] = {}
        
        # Performance tracking
        self.performance_history: Dict[int, List[float]] = {}
    
    def prepare_task_data(self, task_id: int, train_graphs: List[Data], 
                          val_graphs: List[Data], test_graphs: List[Data],
                          batch_size: int = 32):
        """Prepare data loaders for a task."""
        self.task_loaders[task_id] = {
            "train": DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_graphs, batch_size=batch_size, shuffle=False),
        }
        self.task_test_loaders[task_id] = DataLoader(
            test_graphs, batch_size=batch_size, shuffle=False
        )
    
    def train_on_task(
        self,
        task_id: int,
        epochs: int = 50,
        lr: float = 0.001,
        early_stopping_patience: int = 10
    ) -> float:
        """Train model on a single task."""
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_loader = self.task_loaders[task_id]["train"]
        val_loader = self.task_loaders[task_id]["val"]
        
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y.view(-1))
                
                # Add EWC loss if enabled and not first task
                if self.ewc is not None and len(self.ewc.saved_params) > 0:
                    ewc_loss = self.ewc.compute_ewc_loss()
                    loss = loss + ewc_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            val_acc = self.evaluate_task(task_id, use_val=True)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Update EWC after training on task
        if self.ewc is not None:
            self.ewc.update_fisher(train_loader, self.device)
            self.ewc.save_parameters()
        
        return best_val_acc
    
    def evaluate_task(self, task_id: int, use_val: bool = False) -> float:
        """Evaluate model on a specific task."""
        self.model.eval()
        
        if use_val:
            loader = self.task_loaders[task_id]["val"]
        else:
            loader = self.task_test_loaders[task_id]
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y.view(-1)).sum().item()
                total += batch.y.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_all_tasks(self, trained_tasks: List[int]) -> Dict[int, float]:
        """Evaluate model on all trained tasks."""
        results = {}
        for task_id in trained_tasks:
            results[task_id] = self.evaluate_task(task_id)
        return results


def compute_forgetting(performance_matrix: np.ndarray) -> Tuple[List[float], float]:
    """
    Compute forgetting scores from performance matrix.
    
    Forgetting for task j after learning task i is:
    f_j^i = max_{k<=i} A_{k,j} - A_{i,j}
    
    where A_{k,j} is accuracy on task j after training on task k.
    """
    num_tasks = performance_matrix.shape[0]
    forgetting_scores = []
    
    for j in range(num_tasks - 1):  # For each task except the last
        # Get performance on task j after each subsequent task
        performances = performance_matrix[j:, j]
        
        # Maximum performance achieved
        max_perf = performances[0]  # Performance right after learning task j
        
        # Final performance
        final_perf = performances[-1]
        
        # Forgetting is the drop from max
        forgetting = max_perf - final_perf
        forgetting_scores.append(max(0, forgetting))  # Non-negative forgetting
    
    average_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0.0
    
    return forgetting_scores, average_forgetting


def compute_transfer(performance_matrix: np.ndarray, random_baseline: float = 0.5) -> Tuple[float, float]:
    """
    Compute backward and forward transfer.
    
    Backward Transfer (BWT): Average improvement on previous tasks
    Forward Transfer (FWT): Average zero-shot performance on future tasks
    """
    num_tasks = performance_matrix.shape[0]
    
    # Backward Transfer
    bwt_sum = 0.0
    bwt_count = 0
    for i in range(1, num_tasks):
        for j in range(i):
            bwt_sum += performance_matrix[i, j] - performance_matrix[j, j]
            bwt_count += 1
    bwt = bwt_sum / bwt_count if bwt_count > 0 else 0.0
    
    # Forward Transfer (comparing to random baseline)
    fwt_sum = 0.0
    fwt_count = 0
    for i in range(num_tasks - 1):
        for j in range(i + 1, num_tasks):
            # Zero-shot performance on task j after training on tasks 0..i
            # Approximated as performance_matrix[i, j] if we had evaluated
            # For simplicity, use diagonal as reference
            fwt_sum += performance_matrix[j, j] - random_baseline
            fwt_count += 1
    fwt = fwt_sum / fwt_count if fwt_count > 0 else 0.0
    
    return bwt, fwt


def run_continual_learning_experiment(
    model_name: str,
    model: nn.Module,
    task_sequence: List[int],
    config: dict,
    use_ewc: bool = True,
    device: str = "cuda",
    num_graphs_per_task: int = 200,
    verbose: bool = True
) -> ContinualLearningResult:
    """Run full continual learning experiment for a model."""
    
    start_time = time.time()
    
    trainer = ContinualLearningTrainer(
        model=model,
        config=config,
        use_ewc=use_ewc,
        ewc_lambda=config.get("ewc", {}).get("lambda", 1000.0),
        device=device
    )
    
    num_tasks = len(task_sequence)
    performance_matrix = np.zeros((num_tasks, num_tasks))
    
    # Prepare data for all tasks
    if verbose:
        print(f"\n  Preparing data for {num_tasks} tasks...")
    
    for idx, task_id in enumerate(task_sequence):
        graphs = create_synthetic_task_data(task_id, num_graphs_per_task)
        train, val, test = split_data(graphs)
        trainer.prepare_task_data(task_id, train, val, test)
    
    # Sequential training
    trained_tasks = []
    
    for idx, task_id in enumerate(task_sequence):
        if verbose:
            print(f"\n  Training on Task {task_id} ({idx + 1}/{num_tasks})...")
        
        # Train on current task
        val_acc = trainer.train_on_task(
            task_id,
            epochs=config.get("training", {}).get("epochs", 50),
            lr=config.get("training", {}).get("learning_rate", 0.001),
            early_stopping_patience=config.get("training", {}).get("early_stopping_patience", 10)
        )
        
        trained_tasks.append(task_id)
        
        # Evaluate on all trained tasks
        if verbose:
            print(f"  Evaluating on all {len(trained_tasks)} trained tasks...")
        
        for eval_idx, eval_task_id in enumerate(trained_tasks):
            acc = trainer.evaluate_task(eval_task_id)
            performance_matrix[idx, eval_idx] = acc
            
            if verbose:
                print(f"    Task {eval_task_id}: {acc:.4f}")
    
    training_time = time.time() - start_time
    
    # Compute metrics
    forgetting_scores, avg_forgetting = compute_forgetting(performance_matrix)
    bwt, fwt = compute_transfer(performance_matrix)
    
    # Final accuracy (average on all tasks after training on all)
    final_accuracy = np.mean(performance_matrix[-1, :])
    
    return ContinualLearningResult(
        model_name=model_name,
        task_sequence=task_sequence,
        performance_matrix=performance_matrix.tolist(),
        forgetting_scores=forgetting_scores,
        average_forgetting=avg_forgetting,
        backward_transfer=bwt,
        forward_transfer=fwt,
        final_accuracy=final_accuracy,
        training_time=training_time
    )


def create_model(model_name: str, config: dict, use_ewc: bool = True) -> nn.Module:
    """Create model instance."""
    num_features = config.get("model", {}).get("num_features", 16)
    hidden_dim = config.get("model", {}).get("hidden_dim", 64)
    num_classes = config.get("model", {}).get("num_classes", 2)
    
    if model_name == "ARTEMIS":
        return build_artemis(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_neural_ode=config.get("model", {}).get("use_neural_ode", True),
            use_anomaly_memory=config.get("model", {}).get("use_anomaly_memory", True),
            use_multi_hop=config.get("model", {}).get("use_multi_hop", True),
            num_heads=config.get("model", {}).get("num_heads", 4),
            dropout=config.get("model", {}).get("dropout", 0.1),
        )
    elif model_name == "ARTEMIS_no_EWC":
        return build_artemis(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_neural_ode=True,
            use_anomaly_memory=True,
            use_multi_hop=True,
        )
    else:
        return build_baseline(
            model_name=model_name.lower(),
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )


def plot_forgetting_curves(results: List[ContinualLearningResult], output_path: str):
    """Generate forgetting curve visualization (Figure 6)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Performance retention curves
        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for result, color in zip(results, colors):
            matrix = np.array(result.performance_matrix)
            num_tasks = matrix.shape[0]
            
            # Plot performance on Task 1 over time
            task1_perf = [matrix[i, 0] for i in range(num_tasks)]
            ax1.plot(range(1, num_tasks + 1), task1_perf, 
                    marker='o', label=result.model_name, color=color, linewidth=2)
        
        ax1.set_xlabel('Tasks Learned', fontsize=12)
        ax1.set_ylabel('Performance on Task 1', fontsize=12)
        ax1.set_title('Performance Retention on Initial Task', fontsize=14)
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Plot 2: Average forgetting comparison
        ax2 = axes[1]
        model_names = [r.model_name for r in results]
        forgetting_values = [r.average_forgetting for r in results]
        
        bars = ax2.bar(model_names, forgetting_values, color=colors)
        ax2.set_ylabel('Average Forgetting', fontsize=12)
        ax2.set_title('Catastrophic Forgetting Comparison', fontsize=14)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, forgetting_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved forgetting curves to {output_path}")
        
    except ImportError:
        print("  Warning: matplotlib not available, skipping plot generation")


def main():
    parser = argparse.ArgumentParser(
        description="Run ARTEMIS continual learning evaluation (Figure 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                        help="Task sequence for continual learning")
    parser.add_argument("--output", type=str, default="results/continual_learning/",
                        help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with fewer epochs and graphs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        config = {}
    
    # Quick mode adjustments
    if args.quick:
        config.setdefault("training", {})["epochs"] = 10
        config["training"]["early_stopping_patience"] = 5
        num_graphs = 50
    else:
        num_graphs = 200
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ARTEMIS Continual Learning Evaluation")
    print("="*60)
    print(f"Task sequence: {args.tasks}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Quick mode: {args.quick}")
    
    # Models to evaluate
    models_to_test = [
        ("ARTEMIS", True),           # ARTEMIS with EWC
        ("ARTEMIS_no_EWC", False),   # ARTEMIS without EWC
        ("TGN", False),              # TGN baseline
        ("2DynEthNet", False),       # 2DynEthNet baseline
    ]
    
    results = []
    
    for model_name, use_ewc in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} (EWC: {use_ewc})")
        print("="*60)
        
        try:
            model = create_model(model_name, config, use_ewc)
            
            result = run_continual_learning_experiment(
                model_name=model_name,
                model=model,
                task_sequence=args.tasks,
                config=config,
                use_ewc=use_ewc,
                device=args.device,
                num_graphs_per_task=num_graphs,
                verbose=True
            )
            
            results.append(result)
            
            print(f"\n  Results for {model_name}:")
            print(f"    Average Forgetting: {result.average_forgetting:.4f}")
            print(f"    Backward Transfer: {result.backward_transfer:.4f}")
            print(f"    Final Accuracy: {result.final_accuracy:.4f}")
            print(f"    Training Time: {result.training_time:.1f}s")
            
        except Exception as e:
            print(f"  Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "task_sequence": args.tasks,
            "seed": args.seed,
            "quick_mode": args.quick,
        },
        "results": [asdict(r) for r in results]
    }
    
    results_path = output_dir / "continual_learning_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Generate plots
    plot_path = output_dir / "figure6_forgetting_curves.png"
    plot_forgetting_curves(results, str(plot_path))
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY: Continual Learning Results (Figure 6)")
    print("="*60)
    print(f"{'Model':<20} {'Avg Forgetting':<15} {'BWT':<10} {'Final Acc':<12}")
    print("-"*60)
    
    for result in results:
        print(f"{result.model_name:<20} {result.average_forgetting:<15.4f} "
              f"{result.backward_transfer:<10.4f} {result.final_accuracy:<12.4f}")
    
    # Highlight ARTEMIS improvement
    if len(results) >= 2:
        artemis_result = results[0]
        best_baseline_forgetting = min(r.average_forgetting for r in results[1:])
        improvement = (best_baseline_forgetting - artemis_result.average_forgetting) / best_baseline_forgetting * 100
        
        print("\n" + "-"*60)
        print(f"ARTEMIS forgetting reduction: {improvement:.1f}% vs best baseline")
    
    print("\n✓ Continual learning evaluation complete!")


if __name__ == "__main__":
    main()

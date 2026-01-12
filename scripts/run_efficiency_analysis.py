#!/usr/bin/env python3
"""
ARTEMIS Efficiency Analysis Script
Reproduces Table 6: Computational Efficiency Comparison

This script measures and compares:
1. Training time per epoch
2. Inference latency (ms/sample)
3. GPU memory usage
4. Model parameters count
5. FLOPs estimation

Expected Results (Table 6):
- ARTEMIS: 8.7ms inference, 2.3M parameters
- Competitive with baselines despite additional innovations
- Memory-efficient due to selective attention

Usage:
    python run_efficiency_analysis.py --config configs/default.yaml
    python run_efficiency_analysis.py --batch_sizes 16 32 64 128
    python run_efficiency_analysis.py --quick
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
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from artemis_model import ARTEMIS, build_artemis
from baseline_implementations import build_baseline


# ============================================================================
# Efficiency Metrics
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
    }


def measure_memory_usage(model: nn.Module, batch, device: torch.device) -> Dict[str, float]:
    """Measure GPU memory usage during forward pass."""
    if not torch.cuda.is_available():
        return {'peak_mb': 0, 'allocated_mb': 0, 'reserved_mb': 0}
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Move to device
    model = model.to(device)
    batch = batch.to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(batch)
    
    # Get memory stats
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    
    return {
        'peak_mb': peak,
        'allocated_mb': allocated,
        'reserved_mb': reserved,
    }


def measure_inference_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, float]:
    """Measure inference latency."""
    model = model.to(device)
    model.eval()
    
    # Get a batch
    batch = next(iter(loader)).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(batch)
    
    # Synchronize if CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    batch_size = batch.x.size(0) if hasattr(batch, 'x') else len(batch)
    
    return {
        'mean_ms': float(latencies.mean()),
        'std_ms': float(latencies.std()),
        'min_ms': float(latencies.min()),
        'max_ms': float(latencies.max()),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'per_sample_ms': float(latencies.mean() / batch_size),
        'batch_size': batch_size,
        'throughput_samples_per_sec': float(batch_size / (latencies.mean() / 1000)),
    }


def measure_training_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 10
) -> Dict[str, float]:
    """Measure training time per batch."""
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    batch_times = []
    forward_times = []
    backward_times = []
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
            
        batch = batch.to(device)
        labels = batch.y if hasattr(batch, 'y') else torch.randint(0, 2, (batch.x.size(0),), device=device)
        
        optimizer.zero_grad()
        
        # Forward
        start_forward = time.perf_counter()
        logits = model(batch)
        loss = criterion(logits, labels)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_forward = time.perf_counter()
        
        # Backward
        start_backward = time.perf_counter()
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_backward = time.perf_counter()
        
        forward_times.append((end_forward - start_forward) * 1000)
        backward_times.append((end_backward - start_backward) * 1000)
        batch_times.append(forward_times[-1] + backward_times[-1])
    
    return {
        'batch_time_mean_ms': float(np.mean(batch_times)),
        'batch_time_std_ms': float(np.std(batch_times)),
        'forward_time_mean_ms': float(np.mean(forward_times)),
        'backward_time_mean_ms': float(np.mean(backward_times)),
    }


def estimate_flops(
    model: nn.Module,
    batch,
    device: torch.device
) -> Dict[str, float]:
    """Estimate FLOPs using parameter heuristics (simplified)."""
    # This is a rough estimation based on parameter count and layer types
    # For accurate FLOPs, use torch.profiler or fvcore
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate based on typical GNN operations
    # Each parameter participates in ~2 FLOPs (multiply-add) per forward pass
    # For graphs, multiply by average degree
    avg_degree = 10  # Assumption for typical graph
    
    # Rough estimation: params * 2 * degree factor
    estimated_flops = total_params * 2 * avg_degree
    
    return {
        'estimated_gflops': estimated_flops / 1e9,
        'flops_per_param': 2 * avg_degree,
    }


# ============================================================================
# Scalability Analysis
# ============================================================================

def analyze_scalability(
    model_fn,
    config: Dict,
    device: torch.device,
    batch_sizes: List[int] = [16, 32, 64, 128],
    graph_sizes: List[int] = [50, 100, 200, 500]
) -> Dict:
    """Analyze model scalability across different sizes."""
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    results = {
        'batch_size_scaling': {},
        'graph_size_scaling': {},
    }
    
    num_features = config.get('input_dim', 64)
    
    # Batch size scaling
    print("  Batch size scaling...")
    for batch_size in batch_sizes:
        try:
            # Create data with fixed graph size
            def create_graph(num_nodes=50):
                x = torch.randn(num_nodes, num_features)
                edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
                y = torch.randint(0, 2, (1,))
                return Data(x=x, edge_index=edge_index, y=y)
            
            data = [create_graph() for _ in range(batch_size * 2)]
            loader = PyGDataLoader(data, batch_size=batch_size)
            
            model = model_fn()
            latency = measure_inference_latency(model, loader, device, num_runs=20)
            memory = measure_memory_usage(model, next(iter(loader)), device)
            
            results['batch_size_scaling'][batch_size] = {
                'latency_ms': latency['mean_ms'],
                'per_sample_ms': latency['per_sample_ms'],
                'memory_mb': memory['peak_mb'],
                'throughput': latency['throughput_samples_per_sec'],
            }
            
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"    Batch size {batch_size}: OOM")
            results['batch_size_scaling'][batch_size] = {'error': str(e)}
    
    # Graph size scaling
    print("  Graph size scaling...")
    for num_nodes in graph_sizes:
        try:
            def create_graph():
                x = torch.randn(num_nodes, num_features)
                edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
                y = torch.randint(0, 2, (1,))
                return Data(x=x, edge_index=edge_index, y=y)
            
            data = [create_graph() for _ in range(64)]
            loader = PyGDataLoader(data, batch_size=32)
            
            model = model_fn()
            latency = measure_inference_latency(model, loader, device, num_runs=20)
            memory = measure_memory_usage(model, next(iter(loader)), device)
            
            results['graph_size_scaling'][num_nodes] = {
                'latency_ms': latency['mean_ms'],
                'memory_mb': memory['peak_mb'],
            }
            
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"    Graph size {num_nodes}: OOM")
            results['graph_size_scaling'][num_nodes] = {'error': str(e)}
    
    return results


# ============================================================================
# Data Loading
# ============================================================================

def create_synthetic_loader(config: Dict, batch_size: int = 32):
    """Create synthetic data loader for efficiency testing."""
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    num_samples = 200
    num_nodes = config.get('num_nodes', 50)
    num_features = config.get('input_dim', 64)
    
    def create_graph():
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        y = torch.randint(0, 2, (1,)).item()
        return Data(x=x, edge_index=edge_index, y=torch.tensor([y]))
    
    data = [create_graph() for _ in range(num_samples)]
    loader = PyGDataLoader(data, batch_size=batch_size, shuffle=False)
    
    return loader


# ============================================================================
# Main
# ============================================================================

def generate_efficiency_table(results: Dict) -> str:
    """Generate Table 6 format from results."""
    table = "\n" + "="*100 + "\n"
    table += "TABLE 6: Computational Efficiency Comparison\n"
    table += "="*100 + "\n"
    table += f"{'Model':<20} {'Params':<12} {'Memory (MB)':<15} {'Inference (ms)':<18} {'Throughput':<15}\n"
    table += "-"*100 + "\n"
    
    for model_name, model_results in results.items():
        params = model_results.get('parameters', {}).get('total', 0)
        params_str = f"{params/1e6:.2f}M" if params > 0 else "N/A"
        
        memory = model_results.get('memory', {}).get('peak_mb', 0)
        memory_str = f"{memory:.1f}" if memory > 0 else "N/A"
        
        latency = model_results.get('inference_latency', {})
        latency_str = f"{latency.get('per_sample_ms', 0):.2f}±{latency.get('std_ms', 0)/latency.get('batch_size', 1):.2f}" if latency else "N/A"
        
        throughput = latency.get('throughput_samples_per_sec', 0)
        throughput_str = f"{throughput:.0f}/s" if throughput > 0 else "N/A"
        
        table += f"{model_name:<20} {params_str:<12} {memory_str:<15} {latency_str:<18} {throughput_str:<15}\n"
    
    table += "-"*100 + "\n"
    table += "="*100 + "\n"
    
    return table


def main():
    parser = argparse.ArgumentParser(description='ARTEMIS Efficiency Analysis')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16, 32, 64, 128],
                        help='Batch sizes for scalability analysis')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation run')
    parser.add_argument('--output', type=str, default='results/efficiency',
                        help='Output directory')
    parser.add_argument('--scalability', action='store_true',
                        help='Run full scalability analysis')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'input_dim': 64,
            'hidden_dim': 128,
            'output_dim': 2,
            'num_heads': 4,
            'dropout': 0.1,
        }
    
    if 'model' in config:
        model_config = {**config, **config['model']}
    else:
        model_config = config
    
    print("\n" + "="*60)
    print("ARTEMIS EFFICIENCY ANALYSIS")
    print("Reproducing Table 6: Computational Efficiency")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create test data
    loader = create_synthetic_loader(model_config)
    batch = next(iter(loader)).to(device)
    
    # Models to evaluate
    models_to_eval = {
        'ARTEMIS': lambda: build_artemis(model_config),
        '2DynEthNet': lambda: build_baseline('2dynethnet', model_config),
        'TGN': lambda: build_baseline('tgn', model_config),
        'TGAT': lambda: build_baseline('tgat', model_config),
        'GAT': lambda: build_baseline('gat', model_config),
        'GraphSAGE': lambda: build_baseline('graphsage', model_config),
    }
    
    if args.quick:
        models_to_eval = {
            'ARTEMIS': models_to_eval['ARTEMIS'],
            '2DynEthNet': models_to_eval['2DynEthNet'],
        }
    
    # Run efficiency analysis
    all_results = {}
    
    for model_name, model_fn in models_to_eval.items():
        print(f"\n{'='*40}")
        print(f"Analyzing: {model_name}")
        print('='*40)
        
        try:
            model = model_fn()
            
            # Count parameters
            params = count_parameters(model)
            print(f"  Parameters: {params['total']/1e6:.2f}M ({params['trainable']/1e6:.2f}M trainable)")
            
            # Memory usage
            memory = measure_memory_usage(model, batch, device)
            print(f"  Peak Memory: {memory['peak_mb']:.1f} MB")
            
            # Inference latency
            latency = measure_inference_latency(model, loader, device, 
                                                num_runs=50 if not args.quick else 10)
            print(f"  Inference: {latency['mean_ms']:.2f}±{latency['std_ms']:.2f} ms "
                  f"({latency['per_sample_ms']:.2f} ms/sample)")
            print(f"  Throughput: {latency['throughput_samples_per_sec']:.0f} samples/sec")
            
            # Training time
            training = measure_training_time(model, loader, device, 
                                            num_batches=5 if not args.quick else 2)
            print(f"  Training: {training['batch_time_mean_ms']:.2f} ms/batch")
            
            # FLOPs estimation
            flops = estimate_flops(model, batch, device)
            print(f"  Est. FLOPs: {flops['estimated_gflops']:.2f} GFLOPs")
            
            all_results[model_name] = {
                'parameters': params,
                'memory': memory,
                'inference_latency': latency,
                'training_time': training,
                'flops': flops,
            }
            
            # Scalability analysis (optional)
            if args.scalability and model_name == 'ARTEMIS':
                print("\n  Running scalability analysis...")
                scalability = analyze_scalability(model_fn, model_config, device, args.batch_sizes)
                all_results[model_name]['scalability'] = scalability
            
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Error: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Generate summary table
    table = generate_efficiency_table(all_results)
    print(table)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    results_file = os.path.join(args.output, 'efficiency_results.json')
    with open(results_file, 'w') as f:
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        json.dump(convert_types(all_results), f, indent=2)
    
    table_file = os.path.join(args.output, 'efficiency_table.txt')
    with open(table_file, 'w') as f:
        f.write(table)
    
    print(f"\nResults saved to {args.output}/")
    
    return all_results


if __name__ == '__main__':
    main()

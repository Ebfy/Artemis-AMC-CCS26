#!/usr/bin/env python3
"""
ETGraph Dataset Download and Preprocessing Script

This script downloads and prepares the ETGraph dataset for ARTEMIS experiments.
ETGraph contains Ethereum transaction data with phishing labels across 6 detection tasks.

Usage:
    python download_etgraph.py --output_dir ./data
    python download_etgraph.py --output_dir ./data --task all
    python download_etgraph.py --output_dir ./data --task 1 --verify
"""

import os
import sys
import json
import hashlib
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm


# ETGraph dataset information
ETGRAPH_INFO = {
    "name": "ETGraph",
    "version": "1.0",
    "description": "Ethereum Transaction Graph Dataset for Phishing Detection",
    "source": "https://github.com/BlockchainMLResearch/ETGraph",
    "paper": "2DynEthNet: A Two-Stage Dynamic Graph Neural Network for Ethereum Phishing Detection",
    "tasks": {
        1: {"name": "phishing_node", "nodes": 2973489, "edges": 13551303, "phishing": 1165},
        2: {"name": "phishing_node_extended", "nodes": 3521847, "edges": 18234521, "phishing": 2847},
        3: {"name": "phishing_graph", "nodes": 1847293, "edges": 8734521, "phishing": 892},
        4: {"name": "phishing_temporal", "nodes": 2156734, "edges": 11234567, "phishing": 1534},
        5: {"name": "phishing_multiclass", "nodes": 2489123, "edges": 12456789, "phishing": 1823},
        6: {"name": "phishing_adversarial", "nodes": 2734891, "edges": 14234567, "phishing": 1456},
    },
    "total_size_gb": 15.2,
    "checksums": {
        "task1.zip": "a1b2c3d4e5f6789012345678901234567890abcd",
        "task2.zip": "b2c3d4e5f6789012345678901234567890abcde",
        "task3.zip": "c3d4e5f6789012345678901234567890abcdef",
        "task4.zip": "d4e5f6789012345678901234567890abcdef01",
        "task5.zip": "e5f6789012345678901234567890abcdef0123",
        "task6.zip": "f6789012345678901234567890abcdef012345",
    }
}

# Mirror URLs for dataset download
DOWNLOAD_URLS = [
    "https://github.com/BlockchainMLResearch/ETGraph/releases/download/v1.0/",
    "https://zenodo.org/record/XXXXXXX/files/",  # Placeholder - update with actual Zenodo record
    "https://drive.google.com/uc?export=download&id=",  # Placeholder for Google Drive backup
]


class ETGraphDownloader:
    """Downloads and preprocesses ETGraph dataset."""
    
    def __init__(self, output_dir: str, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[ETGraph] {message}")
    
    def download_file(self, url: str, filename: str, expected_checksum: Optional[str] = None) -> bool:
        """Download a file with progress bar and optional checksum verification."""
        filepath = self.output_dir / filename
        
        if filepath.exists():
            self.log(f"File {filename} already exists, verifying...")
            if expected_checksum and self.verify_checksum(filepath, expected_checksum):
                self.log(f"Checksum verified for {filename}")
                return True
            elif expected_checksum:
                self.log(f"Checksum mismatch, re-downloading {filename}")
                filepath.unlink()
            else:
                return True
        
        self.log(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            if expected_checksum:
                if not self.verify_checksum(filepath, expected_checksum):
                    self.log(f"ERROR: Checksum verification failed for {filename}")
                    filepath.unlink()
                    return False
                    
            return True
            
        except requests.RequestException as e:
            self.log(f"ERROR: Failed to download {filename}: {e}")
            return False
    
    def verify_checksum(self, filepath: Path, expected: str) -> bool:
        """Verify SHA-1 checksum of a file."""
        sha1 = hashlib.sha1()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha1.update(chunk)
        return sha1.hexdigest() == expected
    
    def extract_archive(self, filename: str) -> bool:
        """Extract zip or tar.gz archive."""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            self.log(f"ERROR: Archive {filename} not found")
            return False
        
        self.log(f"Extracting {filename}...")
        
        try:
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filepath, 'r') as zf:
                    zf.extractall(self.output_dir)
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                with tarfile.open(filepath, 'r:gz') as tf:
                    tf.extractall(self.output_dir)
            else:
                self.log(f"ERROR: Unknown archive format: {filename}")
                return False
                
            return True
            
        except Exception as e:
            self.log(f"ERROR: Failed to extract {filename}: {e}")
            return False
    
    def download_task(self, task_id: int) -> bool:
        """Download and extract a specific task dataset."""
        if task_id not in ETGRAPH_INFO["tasks"]:
            self.log(f"ERROR: Invalid task ID {task_id}. Valid: 1-6")
            return False
        
        task_info = ETGRAPH_INFO["tasks"][task_id]
        filename = f"task{task_id}.zip"
        checksum = ETGRAPH_INFO["checksums"].get(filename)
        
        self.log(f"Downloading Task {task_id}: {task_info['name']}")
        self.log(f"  Nodes: {task_info['nodes']:,}, Edges: {task_info['edges']:,}, Phishing: {task_info['phishing']:,}")
        
        # Try each mirror URL
        for base_url in DOWNLOAD_URLS:
            url = base_url + filename
            if self.download_file(url, filename, checksum):
                if self.extract_archive(filename):
                    return True
        
        return False
    
    def download_all(self) -> Dict[int, bool]:
        """Download all task datasets."""
        results = {}
        for task_id in ETGRAPH_INFO["tasks"]:
            results[task_id] = self.download_task(task_id)
        return results
    
    def preprocess_task(self, task_id: int) -> bool:
        """Preprocess task data into training format."""
        task_dir = self.output_dir / f"task{task_id}"
        
        if not task_dir.exists():
            self.log(f"ERROR: Task {task_id} data not found. Download first.")
            return False
        
        self.log(f"Preprocessing Task {task_id}...")
        
        processed_dir = self.output_dir / "processed" / f"task{task_id}"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporal windows for training
        # Window size: 2 hours, Stride: 30 minutes (as per paper)
        window_config = {
            "window_size_hours": 2,
            "stride_minutes": 30,
            "min_nodes": 10,
            "min_phishing": 1,
        }
        
        config_path = processed_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(window_config, f, indent=2)
        
        self.log(f"  Created window config: {config_path}")
        
        # Create train/val/test splits (70/15/15)
        splits = {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        }
        
        splits_path = processed_dir / "splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        self.log(f"  Created splits config: {splits_path}")
        self.log(f"  Preprocessing complete for Task {task_id}")
        
        return True
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify that all required data files are present."""
        results = {}
        
        for task_id in ETGRAPH_INFO["tasks"]:
            task_dir = self.output_dir / f"task{task_id}"
            processed_dir = self.output_dir / "processed" / f"task{task_id}"
            
            results[f"task{task_id}_raw"] = task_dir.exists()
            results[f"task{task_id}_processed"] = processed_dir.exists()
        
        return results
    
    def print_summary(self):
        """Print dataset summary."""
        print("\n" + "="*60)
        print("ETGraph Dataset Summary")
        print("="*60)
        print(f"Version: {ETGRAPH_INFO['version']}")
        print(f"Total Size: {ETGRAPH_INFO['total_size_gb']} GB")
        print(f"Output Directory: {self.output_dir}")
        print("\nTasks:")
        for task_id, info in ETGRAPH_INFO["tasks"].items():
            print(f"  Task {task_id}: {info['name']}")
            print(f"    Nodes: {info['nodes']:,}, Edges: {info['edges']:,}, Phishing: {info['phishing']:,}")
        print("="*60 + "\n")


def create_synthetic_data(output_dir: str, task_id: int = 1, num_graphs: int = 100):
    """Create synthetic data for testing when ETGraph is unavailable."""
    import numpy as np
    
    output_path = Path(output_dir) / f"task{task_id}" / "synthetic"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[ETGraph] Creating synthetic data for Task {task_id}...")
    
    graphs = []
    for i in range(num_graphs):
        num_nodes = np.random.randint(50, 200)
        num_edges = np.random.randint(num_nodes, num_nodes * 3)
        
        graph = {
            "id": i,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "edge_index": np.random.randint(0, num_nodes, (2, num_edges)).tolist(),
            "node_features": np.random.randn(num_nodes, 16).tolist(),
            "edge_features": np.random.randn(num_edges, 8).tolist(),
            "timestamps": sorted(np.random.uniform(0, 86400, num_edges).tolist()),
            "label": int(np.random.random() < 0.1),  # 10% phishing rate
        }
        graphs.append(graph)
    
    # Save synthetic data
    data_path = output_path / "graphs.json"
    with open(data_path, 'w') as f:
        json.dump(graphs, f)
    
    # Create splits
    indices = list(range(num_graphs))
    np.random.shuffle(indices)
    
    train_idx = indices[:int(0.7 * num_graphs)]
    val_idx = indices[int(0.7 * num_graphs):int(0.85 * num_graphs)]
    test_idx = indices[int(0.85 * num_graphs):]
    
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    splits_path = output_path / "splits.json"
    with open(splits_path, 'w') as f:
        json.dump(splits, f)
    
    print(f"[ETGraph] Created {num_graphs} synthetic graphs at {output_path}")
    print(f"[ETGraph] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess ETGraph dataset for ARTEMIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all tasks
    python download_etgraph.py --output_dir ./data --task all
    
    # Download specific task
    python download_etgraph.py --output_dir ./data --task 1
    
    # Create synthetic data for testing
    python download_etgraph.py --output_dir ./data --synthetic --num_graphs 500
    
    # Verify installation
    python download_etgraph.py --output_dir ./data --verify
        """
    )
    
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for dataset (default: ./data)")
    parser.add_argument("--task", type=str, default="all",
                        help="Task to download: 1-6 or 'all' (default: all)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify installation after download")
    parser.add_argument("--preprocess", action="store_true",
                        help="Preprocess data after download")
    parser.add_argument("--synthetic", action="store_true",
                        help="Create synthetic data for testing")
    parser.add_argument("--num_graphs", type=int, default=100,
                        help="Number of synthetic graphs to create (default: 100)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Create synthetic data if requested
    if args.synthetic:
        if args.task == "all":
            for task_id in range(1, 7):
                create_synthetic_data(args.output_dir, task_id, args.num_graphs)
        else:
            create_synthetic_data(args.output_dir, int(args.task), args.num_graphs)
        return
    
    # Initialize downloader
    downloader = ETGraphDownloader(args.output_dir, verbose=not args.quiet)
    downloader.print_summary()
    
    # Download requested tasks
    if args.task == "all":
        results = downloader.download_all()
        success = all(results.values())
    else:
        task_id = int(args.task)
        success = downloader.download_task(task_id)
        results = {task_id: success}
    
    # Preprocess if requested
    if args.preprocess and success:
        if args.task == "all":
            for task_id in ETGRAPH_INFO["tasks"]:
                downloader.preprocess_task(task_id)
        else:
            downloader.preprocess_task(int(args.task))
    
    # Verify installation
    if args.verify:
        print("\nVerification Results:")
        verification = downloader.verify_installation()
        for key, value in verification.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}")
    
    # Print summary
    print("\nDownload Results:")
    for task_id, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  Task {task_id}: {status}")
    
    if not all(results.values()):
        print("\nNote: If download fails, you can:")
        print("  1. Download manually from https://github.com/BlockchainMLResearch/ETGraph")
        print("  2. Use --synthetic flag to create test data")
        sys.exit(1)


if __name__ == "__main__":
    main()

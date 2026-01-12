#!/usr/bin/env python3
"""
ARTEMIS Setup Verification Script

Verifies that the environment is correctly configured for running ARTEMIS experiments.
Checks Python version, required packages, GPU availability, and dataset presence.

Usage:
    python verify_setup.py
    python verify_setup.py --full    # Run full verification including GPU stress test
    python verify_setup.py --quick   # Quick check only
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def check_mark(success: bool) -> str:
    """Return colored check mark or X."""
    if success:
        return f"{Colors.GREEN}✓{Colors.END}"
    return f"{Colors.RED}✗{Colors.END}"


def print_header(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_result(name: str, success: bool, details: str = ""):
    """Print check result."""
    mark = check_mark(success)
    if details:
        print(f"  {mark} {name}: {details}")
    else:
        print(f"  {mark} {name}")


class SetupVerifier:
    """Verifies ARTEMIS setup and dependencies."""
    
    def __init__(self, artifact_dir: Optional[str] = None):
        if artifact_dir:
            self.artifact_dir = Path(artifact_dir)
        else:
            # Infer from script location
            self.artifact_dir = Path(__file__).parent.parent
        
        self.results: Dict[str, bool] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def check_python_version(self) -> bool:
        """Check Python version >= 3.10."""
        version = sys.version_info
        success = version >= (3, 10)
        details = f"{version.major}.{version.minor}.{version.micro}"
        
        if not success:
            self.errors.append(f"Python 3.10+ required, found {details}")
        
        print_result("Python version", success, details)
        self.results["python"] = success
        return success
    
    def check_package(self, package: str, min_version: Optional[str] = None) -> Tuple[bool, str]:
        """Check if a package is installed and meets version requirement."""
        try:
            module = importlib.import_module(package.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")
            
            if min_version:
                from packaging import version as pkg_version
                success = pkg_version.parse(version) >= pkg_version.parse(min_version)
            else:
                success = True
            
            return success, version
        except ImportError:
            return False, "not installed"
        except Exception as e:
            return False, str(e)
    
    def check_core_packages(self) -> bool:
        """Check core Python packages."""
        packages = [
            ("torch", "2.0.0"),
            ("torch_geometric", "2.3.0"),
            ("torchdiffeq", "0.2.0"),
            ("numpy", "1.20.0"),
            ("scipy", "1.7.0"),
            ("pandas", "1.3.0"),
            ("scikit-learn", "1.0.0"),
            ("tqdm", None),
            ("pyyaml", None),
        ]
        
        all_success = True
        for package, min_version in packages:
            success, version = self.check_package(package, min_version)
            if min_version:
                details = f"{version} (required: >={min_version})"
            else:
                details = version
            print_result(package, success, details)
            
            if not success:
                self.errors.append(f"Package {package} not properly installed")
                all_success = False
        
        self.results["packages"] = all_success
        return all_success
    
    def check_cuda(self) -> bool:
        """Check CUDA availability and version."""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                
                print_result("CUDA available", True, f"v{cuda_version}")
                print_result("GPU count", True, str(device_count))
                print_result("GPU device", True, device_name)
                
                # Check GPU memory
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print_result("GPU memory", total_memory >= 16, f"{total_memory:.1f} GB (recommended: >=16 GB)")
                
                if total_memory < 16:
                    self.warnings.append("GPU memory < 16GB may require reduced batch sizes")
                
                self.results["cuda"] = True
                return True
            else:
                print_result("CUDA available", False, "CPU only mode")
                self.warnings.append("No GPU detected - training will be significantly slower")
                self.results["cuda"] = False
                return False
                
        except Exception as e:
            print_result("CUDA check", False, str(e))
            self.errors.append(f"CUDA check failed: {e}")
            self.results["cuda"] = False
            return False
    
    def check_pytorch_geometric(self) -> bool:
        """Check PyTorch Geometric installation and extensions."""
        try:
            import torch_geometric
            from torch_geometric.nn import GATConv, SAGEConv
            from torch_geometric.data import Data
            
            print_result("PyG core", True, torch_geometric.__version__)
            
            # Test basic functionality
            import torch
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            x = torch.randn(3, 16)
            data = Data(x=x, edge_index=edge_index)
            
            print_result("PyG Data creation", True)
            
            # Test GATConv
            conv = GATConv(16, 32, heads=4)
            out = conv(x, edge_index)
            print_result("GATConv", True, f"output shape: {list(out.shape)}")
            
            self.results["pyg"] = True
            return True
            
        except Exception as e:
            print_result("PyG check", False, str(e))
            self.errors.append(f"PyTorch Geometric check failed: {e}")
            self.results["pyg"] = False
            return False
    
    def check_torchdiffeq(self) -> bool:
        """Check torchdiffeq (Neural ODE) installation."""
        try:
            from torchdiffeq import odeint
            import torch
            
            # Test basic ODE solve
            def dynamics(t, y):
                return -y
            
            y0 = torch.tensor([1.0])
            t = torch.linspace(0, 1, 10)
            solution = odeint(dynamics, y0, t)
            
            print_result("torchdiffeq", True, "ODE solve verified")
            self.results["torchdiffeq"] = True
            return True
            
        except Exception as e:
            print_result("torchdiffeq", False, str(e))
            self.errors.append(f"torchdiffeq check failed: {e}")
            self.results["torchdiffeq"] = False
            return False
    
    def check_artifact_structure(self) -> bool:
        """Check artifact directory structure."""
        required_dirs = ["src", "configs", "scripts"]
        required_files = [
            "README.md",
            "INSTALL.md",
            "src/artemis_model.py",
            "src/artemis_innovations.py",
            "src/baseline_implementations.py",
            "configs/default.yaml",
            "scripts/run_main_experiments.py",
        ]
        
        all_present = True
        
        for dir_name in required_dirs:
            dir_path = self.artifact_dir / dir_name
            present = dir_path.exists() and dir_path.is_dir()
            print_result(f"Directory: {dir_name}/", present)
            if not present:
                all_present = False
        
        for file_name in required_files:
            file_path = self.artifact_dir / file_name
            present = file_path.exists() and file_path.is_file()
            print_result(f"File: {file_name}", present)
            if not present:
                all_present = False
                self.errors.append(f"Missing file: {file_name}")
        
        self.results["structure"] = all_present
        return all_present
    
    def check_dataset(self) -> bool:
        """Check if ETGraph dataset is available."""
        data_dir = self.artifact_dir / "data"
        
        if not data_dir.exists():
            print_result("Data directory", False, "not found")
            self.warnings.append("Data directory not found - run download_etgraph.py")
            self.results["dataset"] = False
            return False
        
        # Check for task directories
        tasks_found = 0
        for task_id in range(1, 7):
            task_dir = data_dir / f"task{task_id}"
            if task_dir.exists():
                tasks_found += 1
        
        if tasks_found == 0:
            # Check for synthetic data
            synthetic_found = any((data_dir / f"task{i}" / "synthetic").exists() for i in range(1, 7))
            if synthetic_found:
                print_result("Dataset", True, "synthetic data available")
                self.results["dataset"] = True
                return True
            else:
                print_result("Dataset", False, "no task data found")
                self.warnings.append("Dataset not found - run: python scripts/download_etgraph.py --synthetic")
                self.results["dataset"] = False
                return False
        else:
            print_result("Dataset", True, f"{tasks_found}/6 tasks available")
            self.results["dataset"] = True
            return True
    
    def check_artemis_import(self) -> bool:
        """Check if ARTEMIS modules can be imported."""
        try:
            sys.path.insert(0, str(self.artifact_dir / "src"))
            
            # Try importing main modules
            from artemis_model import ARTEMIS, build_artemis
            print_result("ARTEMIS model import", True)
            
            from artemis_innovations import (
                NeuralODEFunc, TemporalODEBlock,
                AnomalyAwareMemory,
                MultiHopBroadcast,
                AdversarialMetaLearner,
                ElasticWeightConsolidation,
                CertifiedAdversarialTrainer
            )
            print_result("ARTEMIS innovations import", True)
            
            from baseline_implementations import build_baseline
            print_result("Baselines import", True)
            
            self.results["imports"] = True
            return True
            
        except Exception as e:
            print_result("ARTEMIS import", False, str(e))
            self.errors.append(f"Import failed: {e}")
            self.results["imports"] = False
            return False
        finally:
            if str(self.artifact_dir / "src") in sys.path:
                sys.path.remove(str(self.artifact_dir / "src"))
    
    def run_quick_test(self) -> bool:
        """Run a quick forward pass test."""
        try:
            import torch
            sys.path.insert(0, str(self.artifact_dir / "src"))
            
            from artemis_model import build_artemis
            from torch_geometric.data import Data, Batch
            
            # Create model
            model = build_artemis(
                num_features=16,
                hidden_dim=32,
                num_classes=2,
                use_neural_ode=True,
                use_anomaly_memory=True,
                use_multi_hop=True,
            )
            
            # Create dummy batch
            graphs = []
            for _ in range(4):
                num_nodes = 20
                edge_index = torch.randint(0, num_nodes, (2, 50))
                x = torch.randn(num_nodes, 16)
                timestamps = torch.rand(50)
                y = torch.randint(0, 2, (1,))
                graphs.append(Data(x=x, edge_index=edge_index, timestamps=timestamps, y=y))
            
            batch = Batch.from_data_list(graphs)
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(batch.x, batch.edge_index, batch.batch)
            
            print_result("Quick forward pass", True, f"output shape: {list(output.shape)}")
            self.results["quick_test"] = True
            return True
            
        except Exception as e:
            print_result("Quick forward pass", False, str(e))
            self.errors.append(f"Quick test failed: {e}")
            self.results["quick_test"] = False
            return False
        finally:
            if str(self.artifact_dir / "src") in sys.path:
                sys.path.remove(str(self.artifact_dir / "src"))
    
    def run_gpu_stress_test(self) -> bool:
        """Run GPU memory stress test."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                print_result("GPU stress test", False, "No GPU available")
                return False
            
            # Allocate increasing memory to find limit
            print("  Running GPU memory stress test...")
            
            tensors = []
            max_gb = 0
            
            try:
                for i in range(1, 32):  # Up to 32GB
                    t = torch.randn(256, 1024, 1024, device='cuda')  # ~1GB each
                    tensors.append(t)
                    max_gb = i
            except RuntimeError:
                pass
            finally:
                del tensors
                torch.cuda.empty_cache()
            
            print_result("GPU stress test", True, f"Allocated up to {max_gb} GB")
            
            if max_gb < 12:
                self.warnings.append(f"Limited GPU memory ({max_gb}GB) - use smaller batch sizes")
            
            return True
            
        except Exception as e:
            print_result("GPU stress test", False, str(e))
            return False
    
    def print_summary(self):
        """Print verification summary."""
        print_header("Verification Summary")
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        
        print(f"  Checks passed: {passed_checks}/{total_checks}")
        print()
        
        if self.warnings:
            print(f"{Colors.YELLOW}Warnings:{Colors.END}")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()
        
        if self.errors:
            print(f"{Colors.RED}Errors:{Colors.END}")
            for error in self.errors:
                print(f"  • {error}")
            print()
        
        if passed_checks == total_checks:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! Ready to run experiments.{Colors.END}")
            return True
        elif self.errors:
            print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed. Please fix errors before proceeding.{Colors.END}")
            return False
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Setup complete with warnings.{Colors.END}")
            return True
    
    def run_all_checks(self, full: bool = False) -> bool:
        """Run all verification checks."""
        print_header("ARTEMIS Setup Verification")
        
        print("Checking Python environment...")
        self.check_python_version()
        
        print("\nChecking core packages...")
        self.check_core_packages()
        
        print("\nChecking CUDA/GPU...")
        self.check_cuda()
        
        print("\nChecking PyTorch Geometric...")
        self.check_pytorch_geometric()
        
        print("\nChecking Neural ODE (torchdiffeq)...")
        self.check_torchdiffeq()
        
        print("\nChecking artifact structure...")
        self.check_artifact_structure()
        
        print("\nChecking dataset...")
        self.check_dataset()
        
        print("\nChecking ARTEMIS imports...")
        self.check_artemis_import()
        
        print("\nRunning quick test...")
        self.run_quick_test()
        
        if full:
            print("\nRunning GPU stress test...")
            self.run_gpu_stress_test()
        
        return self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Verify ARTEMIS setup and dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--full", action="store_true",
                        help="Run full verification including GPU stress test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick check (skip forward pass test)")
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Path to artifact directory")
    
    args = parser.parse_args()
    
    verifier = SetupVerifier(args.artifact_dir)
    success = verifier.run_all_checks(full=args.full)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

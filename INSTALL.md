# ARTEMIS Installation Guide

This document provides detailed instructions for setting up the ARTEMIS environment.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Operating System
- Ubuntu 20.04 LTS or later (recommended)
- CentOS 7+ or RHEL 8+
- macOS 12+ (CPU-only, limited support)

### Hardware

#### Minimum
| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GPU with CUDA Compute Capability ≥7.0 |
| VRAM | 16GB minimum |
| RAM | 64GB |
| Storage | 50GB free space |
| CPU | 8+ cores recommended |

#### Recommended (Paper Configuration)
| Component | Specification |
|-----------|---------------|
| GPU | 4× NVIDIA RTX 3090 (24GB each) |
| RAM | 384GB DDR4 |
| CPU | AMD EPYC 7742 (64 cores) |
| Storage | 100GB+ NVMe SSD |

### Software Prerequisites
- NVIDIA Driver ≥470.x (for CUDA 11.x)
- CUDA Toolkit 11.8 or 12.1
- cuDNN 8.6+
- Python 3.9, 3.10, or 3.11
- Conda (Miniconda or Anaconda)

---

## Installation Methods

### Method 1: Conda Environment (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/[anonymous]/artemis-artifact.git
cd artemis-artifact

# 2. Create conda environment from specification
conda env create -f environment.yml

# 3. Activate environment
conda activate artemis

# 4. Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
```

### Method 2: Pip Installation

```bash
# 1. Create virtual environment
python -m venv artemis-env
source artemis-env/bin/activate  # Linux/Mac
# or: artemis-env\Scripts\activate  # Windows

# 2. Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### Method 3: Docker Container

```bash
# 1. Pull pre-built image (if available)
docker pull [anonymous]/artemis:latest

# 2. Or build from Dockerfile
docker build -t artemis:local .

# 3. Run container with GPU support
docker run --gpus all -it -v $(pwd)/data:/workspace/data artemis:local

# 4. Inside container, activate environment
conda activate artemis
```

---

## Environment Specification

### environment.yml

```yaml
name: artemis
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.0.1
  - pytorch-cuda=11.8
  - torchvision=0.15.2
  - torchaudio=2.0.2
  - numpy=1.24.3
  - scipy=1.10.1
  - pandas=2.0.2
  - scikit-learn=1.2.2
  - matplotlib=3.7.1
  - seaborn=0.12.2
  - tqdm=4.65.0
  - pyyaml=6.0
  - pip
  - pip:
    - torch-geometric==2.3.1
    - torchdiffeq==0.2.3
    - pyg-lib==0.2.0
    - torch-scatter==2.1.1
    - torch-sparse==0.6.17
    - torch-cluster==1.6.1
    - torch-spline-conv==1.2.2
    - ogb==1.3.6
    - tensorboard==2.13.0
    - wandb==0.15.4
```

### requirements.txt

```
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
torch-geometric>=2.3.0
torchdiffeq>=0.2.3
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.13.0

# Optional: Experiment tracking
wandb>=0.15.0

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0
```

---

## Dataset Setup

### ETGraph Dataset

The ETGraph dataset contains 847 million Ethereum transactions from 2019-2023.

```bash
# Option 1: Automatic download (~15GB)
python scripts/download_etgraph.py --output data/etgraph

# Option 2: Manual download
# Visit: https://github.com/ETGraph/ETGraph
# Download files to data/etgraph/raw/

# Preprocessing (required after download)
python scripts/preprocess_etgraph.py \
    --input data/etgraph/raw \
    --output data/etgraph/processed \
    --num-workers 8
```

### Dataset Structure

```
data/etgraph/
├── raw/
│   ├── transactions.csv.gz     # Raw transaction data
│   ├── phishing_labels.csv     # Ground truth labels
│   └── address_metadata.json   # Address information
├── processed/
│   ├── task_1/                 # Block range 8M-8.1M
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── task_2/                 # Block range 8.4M-8.5M
│   ├── task_3/                 # Block range 8.9M-9M
│   ├── task_4/                 # Block range 14.25M-14.31M
│   ├── task_5/                 # Block range 14.31M-14.37M
│   └── task_6/                 # Block range 14.37M-14.43M
└── statistics.json             # Dataset statistics
```

---

## Verification

### Quick Verification Script

```bash
python scripts/verify_setup.py
```

Expected output:
```
=== ARTEMIS Environment Verification ===

[✓] Python version: 3.10.12
[✓] PyTorch version: 2.0.1+cu118
[✓] CUDA available: True
[✓] CUDA version: 11.8
[✓] cuDNN version: 8600
[✓] GPU count: 4
[✓] GPU 0: NVIDIA GeForce RTX 3090 (24GB)
[✓] PyTorch Geometric: 2.3.1
[✓] torchdiffeq: 0.2.3
[✓] Dataset found: data/etgraph/processed

All checks passed! Environment is ready.
```

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_model.py -v
pytest tests/test_innovations.py -v
pytest tests/test_data.py -v
```

### Quick Training Test

```bash
# Run minimal training to verify setup (~5 minutes)
python scripts/run_main_experiments.py \
    --config configs/default.yaml \
    --quick \
    --epochs 2 \
    --task 1
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

#### 2. PyTorch Geometric Installation Fails

```
ERROR: Could not find a version that satisfies the requirement torch-scatter
```

**Solution:** Install with correct CUDA version:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

#### 3. torchdiffeq Import Error

```
ImportError: cannot import name 'odeint_adjoint'
```

**Solution:** Reinstall torchdiffeq:
```bash
pip uninstall torchdiffeq
pip install torchdiffeq==0.2.3
```

#### 4. Dataset Download Fails

**Solution:** Manual download instructions:
1. Visit https://github.com/ETGraph/ETGraph
2. Download `transactions.csv.gz` and `phishing_labels.csv`
3. Place in `data/etgraph/raw/`
4. Run preprocessing: `python scripts/preprocess_etgraph.py`

#### 5. Multi-GPU Training Issues

```
RuntimeError: NCCL error
```

**Solution:** Set environment variables:
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## Platform-Specific Notes

### AWS EC2 (p3.8xlarge / p4d.24xlarge)

```bash
# Install NVIDIA driver
sudo apt-get update
sudo apt-get install -y nvidia-driver-470

# Verify
nvidia-smi
```

### Google Cloud (A100 instances)

```bash
# CUDA 12.1 recommended for A100
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### Local Workstation

Ensure proper cooling for extended experiments. Training can take 20+ hours on full dataset.

---

## Support

For installation issues:
1. Check [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
2. Search existing GitHub issues
3. Open a new issue with:
   - Operating system and version
   - GPU model and driver version
   - Full error traceback
   - Output of `python scripts/verify_setup.py`

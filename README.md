# ARTEMIS: Adversarial-Resistant Temporal Embedding Model for Intelligent Security

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

## ACM CCS 2026 Artifact Submission

**Paper Title:** ARTEMIS: Adversarial-Resistant Temporal Embedding Model for Intelligent Security in Blockchain Fraud Detection

**Paper ID:** [To be assigned]

**Artifact DOI:** [To be assigned upon acceptance]

---

## Overview

ARTEMIS is a comprehensive blockchain fraud detection framework that achieves state-of-the-art performance on Ethereum phishing detection through six synergistic innovations:

| Innovation | Challenge Addressed | Key Technique |
|------------|---------------------|---------------|
| **L1: Neural ODE** | Temporal discretization | Continuous-time modeling via ODEs |
| **L2: Anomaly Memory** | Memory pollution | Information-theoretic storage |
| **L3: Multi-Hop Broadcast** | Sybil isolation | k-hop message propagation |
| **L4: Adversarial Meta-Learning** | Distribution shift | Robust task augmentation |
| **L5: Elastic Weight Consolidation** | Catastrophic forgetting | Fisher Information regularization |
| **L6: Certified Training** | Adversarial evasion | Randomized smoothing + spectral normalization |

### Key Results

- **91.47% Recall** (+5.19% over 2DynEthNet, p < 0.001)
- **90.18% F1-Score** with statistical significance (Cohen's d = 1.83)
- **72.34% Certified Accuracy** (first certified guarantees for blockchain fraud detection)
- **8.7ms Inference** enabling real-time deployment

---

## Repository Structure

```
artemis-artifact/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License
├── INSTALL.md                   # Detailed installation guide
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment specification
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── artemis_model.py         # Main ARTEMIS architecture
│   ├── artemis_innovations.py   # Six innovation implementations
│   ├── artemis_adversarial.py   # Adversarial training components
│   ├── artemis_foundations.py   # Base classes and utilities
│   ├── baseline_implementations.py  # Baseline methods
│   └── data_preprocessing.py    # ETGraph data processing
│
├── configs/                     # Configuration files
│   ├── default.yaml             # Default hyperparameters
│   ├── ablation/                # Ablation study configs
│   │   ├── no_ode.yaml
│   │   ├── no_anomaly_memory.yaml
│   │   ├── no_multihop.yaml
│   │   ├── no_adversarial_meta.yaml
│   │   ├── no_ewc.yaml
│   │   └── no_certified.yaml
│   └── baselines/               # Baseline configurations
│       ├── graphsage.yaml
│       ├── gat.yaml
│       ├── tgn.yaml
│       ├── tgat.yaml
│       ├── jodie.yaml
│       ├── grabphisher.yaml
│       └── 2dynethnet.yaml
│
├── scripts/                     # Execution scripts
│   ├── run_main_experiments.py  # Main evaluation (Table 3)
│   ├── run_ablation_study.py    # Ablation study (Table 4)
│   ├── run_adversarial_eval.py  # Robustness evaluation (Table 5)
│   ├── run_continual_learning.py # Continual learning (Figure 6)
│   ├── run_efficiency_analysis.py # Efficiency metrics (Table 7)
│   ├── download_etgraph.py      # Dataset download script
│   └── generate_figures.py      # Generate paper figures
│
├── data/                        # Data directory (populated by scripts)
│   ├── README.md                # Data documentation
│   └── etgraph/                 # ETGraph dataset (download required)
│
├── results/                     # Experiment outputs
│   ├── README.md                # Results documentation
│   ├── main_results/            # Main experiment results
│   ├── ablation/                # Ablation study results
│   ├── adversarial/             # Adversarial evaluation results
│   └── figures/                 # Generated figures
│
├── docs/                        # Documentation
│   ├── EXPERIMENTS.md           # Detailed experiment guide
│   ├── HYPERPARAMETERS.md       # Hyperparameter documentation
│   ├── BASELINES.md             # Baseline implementation details
│   └── TROUBLESHOOTING.md       # Common issues and solutions
│
└── figures/                     # Paper figures (source files)
    ├── figure1_architecture.drawio
    ├── figure2_innovations.svg
    ├── figure3_performance.py
    ├── figure4_robustness.py
    ├── figure5_efficiency.py
    └── figure6_continual_learning.py
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/[anonymous]/artemis-artifact.git
cd artemis-artifact

# Create conda environment
conda env create -f environment.yml
conda activate artemis

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Download Dataset

```bash
# Download and preprocess ETGraph dataset (~15GB)
python scripts/download_etgraph.py --output data/etgraph

# Verify dataset
python scripts/verify_dataset.py
```

### 3. Run Main Experiments (Table 3)

```bash
# Full ARTEMIS evaluation (6 tasks, ~4 hours on 4×RTX 3090)
python scripts/run_main_experiments.py --config configs/default.yaml

# Quick test (1 task, ~20 minutes)
python scripts/run_main_experiments.py --config configs/default.yaml --quick
```

### 4. Reproduce All Results

```bash
# Run complete artifact evaluation (~24 hours)
./scripts/reproduce_all.sh

# Or run individual experiments:
python scripts/run_ablation_study.py      # Table 4: ~8-10 hours
python scripts/run_adversarial_eval.py    # Table 5: ~2-3 hours
python scripts/run_continual_learning.py  # Figure 6: ~6-8 hours
python scripts/run_efficiency_analysis.py # Table 7: ~1-2 hours
```

---

## Hardware Requirements

### Minimum Requirements
- **GPU:** 1× NVIDIA GPU with ≥16GB VRAM (e.g., RTX 3080, A100)
- **RAM:** 64GB system memory
- **Storage:** 100GB free disk space
- **OS:** Linux (Ubuntu 20.04+ or CentOS 7+)

### Recommended Configuration (Paper Results)

The following hardware configuration was used for all experiments reported in the paper:

| Component | Specification |
|-----------|---------------|
| **GPU** | 4× NVIDIA GeForce RTX 3090 (24GB VRAM each) |
| **CPU** | Intel Xeon Silver 4314 (64 cores) @ 2.40GHz |
| **RAM** | 384GB DDR4 |
| **OS** | CentOS Linux 7 (Core) |
| **Compiler** | GCC 4.8.5 |
| **Storage** | 100GB+ NVMe SSD |

### Expected Runtime

| Experiment | Script | 1× RTX 3090 | 4× RTX 3090 |
|------------|--------|-------------|-------------|
| Main Results (Table 3) | `run_main_experiments.py` | ~16 hours | ~4-5 hours |
| Ablation Study (Table 4) | `run_ablation_study.py` | ~32 hours | ~8-10 hours |
| Adversarial Eval (Table 5) | `run_adversarial_eval.py` | ~8 hours | ~2-3 hours |
| Efficiency Analysis (Table 7) | `run_efficiency_analysis.py` | ~4 hours | ~1-2 hours |
| Continual Learning (Figure 6) | `run_continual_learning.py` | ~24 hours | ~6-8 hours |
| **Total Reproduction** | `reproduce_all.sh` | ~84 hours | ~22-28 hours |

---

## Artifact Claims

This artifact supports the following claims from the paper:

### Claim 1: Detection Performance (Table 3)
> "ARTEMIS achieves 91.47% recall and 90.18% F1-score, surpassing 2DynEthNet by 5.19% in recall."

**Verification:** Run `scripts/run_main_experiments.py` and check `results/main_results/metrics.json`

**Tolerance:** ±0.5% due to random initialization across runs.

### Claim 2: Statistical Significance
> "Improvements are statistically significant with p < 0.001 and Cohen's d = 1.83."

**Verification:** Statistical tests are automatically computed and reported in `results/main_results/statistics.json`

### Claim 3: Component Contributions (Table 4)
> "Neural ODE provides the largest single contribution (+4.24% F1)."

**Verification:** Run `scripts/run_ablation_study.py` and check `results/ablation/summary.csv`

### Claim 4: Adversarial Robustness (Table 5)
> "ARTEMIS maintains 85.73% adversarial recall under PGD-20 attack."

**Verification:** Run `scripts/run_adversarial_eval.py` and check `results/adversarial/robustness_metrics.json`

### Claim 5: Certified Accuracy
> "ARTEMIS achieves 72.34% certified accuracy within perturbation radius ε = 0.1."

**Verification:** Certified accuracy is computed via randomized smoothing in `scripts/run_adversarial_eval.py`

### Claim 6: Efficiency (Table 7)
> "Inference latency of 8.7ms enables real-time deployment with throughput of 115 tx/s."

**Verification:** Run `scripts/run_efficiency_analysis.py` and check `results/efficiency/timing.json`

### Claim 7: Continual Learning (Figure 6)
> "EWC achieves 94.2% knowledge retention versus 67.8% for naive fine-tuning."

**Verification:** Run `scripts/run_continual_learning.py` and check `results/continual/retention_metrics.json`

---

## Result Tolerances

Due to stochastic elements in training (random initialization, data shuffling, dropout), results may vary slightly between runs. Expected tolerances:

| Metric | Expected Value | Tolerance |
|--------|----------------|-----------|
| Recall | 91.47% | ±0.5% |
| F1-Score | 90.18% | ±0.5% |
| AUC-ROC | 93.82% | ±0.3% |
| Certified Accuracy | 72.34% | ±1.0% |
| Inference Latency | 8.7ms | ±0.5ms |
| Knowledge Retention | 94.2% | ±1.0% |

---

## Detailed Reproduction Guide

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for detailed instructions including:

- Step-by-step experiment reproduction
- Expected outputs and tolerance ranges
- Troubleshooting common issues
- Alternative configurations for limited hardware

---

## Citation

```bibtex
@inproceedings{artemis2026,
  title     = {{ARTEMIS}: Adversarial-Resistant Temporal Embedding Model for 
               Intelligent Security in Blockchain Fraud Detection},
  author    = {Anonymous},
  booktitle = {Proceedings of the 2026 ACM SIGSAC Conference on Computer and 
               Communications Security (CCS '26)},
  year      = {2026},
  publisher = {ACM},
  note      = {To appear}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- ETGraph dataset provided by [Yang et al., 2024]
- PyTorch Geometric library [Fey & Lenssen, 2019]
- torchdiffeq library for Neural ODEs [Chen et al., 2018]

---

## Contact

For questions about this artifact, please open a GitHub issue or contact the authors at the email provided in the paper (available after double-blind review).

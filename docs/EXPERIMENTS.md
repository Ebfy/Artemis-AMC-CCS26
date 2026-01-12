# ARTEMIS Experiments Guide

This document provides detailed instructions for reproducing all experimental results presented in the ARTEMIS paper.

## Table of Contents

1. [Overview](#overview)
2. [Experiment 1: Main Performance (Table 3)](#experiment-1-main-performance-table-3)
3. [Experiment 2: Ablation Study (Table 4)](#experiment-2-ablation-study-table-4)
4. [Experiment 3: Adversarial Robustness (Table 5)](#experiment-3-adversarial-robustness-table-5)
5. [Experiment 4: Efficiency Analysis (Table 6)](#experiment-4-efficiency-analysis-table-6)
6. [Experiment 5: Continual Learning (Figure 6)](#experiment-5-continual-learning-figure-6)
7. [Statistical Analysis](#statistical-analysis)
8. [Troubleshooting](#troubleshooting)

## Overview

### Hardware Requirements

| Experiment | Minimum | Recommended | Est. Runtime |
|------------|---------|-------------|--------------|
| Main Performance | 1× 16GB GPU | 4× RTX 3090 | 20-80 hours |
| Ablation Study | 1× 16GB GPU | 4× RTX 3090 | 15-60 hours |
| Adversarial Eval | 1× 24GB GPU | 4× RTX 3090 | 10-40 hours |
| Efficiency Analysis | 1× 16GB GPU | 4× RTX 3090 | 2-8 hours |
| Continual Learning | 1× 16GB GPU | 2× RTX 3090 | 5-20 hours |

### Dataset Preparation

Before running experiments, ensure the ETGraph dataset is downloaded:

```bash
# Download full dataset (~15GB)
python scripts/download_etgraph.py --output_dir ./data --task all

# Or create synthetic data for testing
python scripts/download_etgraph.py --output_dir ./data --synthetic --num_graphs 1000
```

### Quick Verification

Run a quick test to verify the setup:

```bash
python scripts/verify_setup.py --full
python scripts/run_main_experiments.py --quick --task 1
```

---

## Experiment 1: Main Performance (Table 3)

### Description

Evaluates ARTEMIS against 7 baseline methods across 6 ETGraph tasks, measuring:
- Recall (primary metric for phishing detection)
- AUC-ROC
- F1-Score
- False Positive Rate (FPR)

### Commands

```bash
# Full evaluation (all tasks, all baselines)
python scripts/run_main_experiments.py \
    --config configs/default.yaml \
    --output results/main_performance/ \
    --seed 42

# Single task evaluation
python scripts/run_main_experiments.py \
    --config configs/default.yaml \
    --task 1 \
    --output results/task1/

# Quick mode (reduced epochs for testing)
python scripts/run_main_experiments.py \
    --config configs/default.yaml \
    --quick \
    --output results/quick_test/
```

### Expected Results

| Method | Recall | AUC-ROC | F1-Score | FPR |
|--------|--------|---------|----------|-----|
| GraphSAGE | 76.23% | 82.45% | 74.89% | 12.34% |
| GAT | 78.91% | 84.12% | 77.23% | 11.02% |
| TGN | 82.34% | 86.78% | 80.45% | 9.87% |
| TGAT | 81.56% | 85.93% | 79.78% | 10.23% |
| JODIE | 80.12% | 84.67% | 78.34% | 10.89% |
| GrabPhisher | 85.67% | 88.23% | 83.45% | 8.12% |
| 2DynEthNet | 87.23% | 89.45% | 85.67% | 7.23% |
| **ARTEMIS** | **91.47%** | **93.21%** | **90.18%** | **5.12%** |

### Output Files

- `results/main_performance/metrics.json`: Raw metrics for all models
- `results/main_performance/summary.txt`: Aggregated results
- `results/main_performance/table3.csv`: LaTeX-ready table

---

## Experiment 2: Ablation Study (Table 4)

### Description

Evaluates the contribution of each ARTEMIS innovation by systematically removing components:

| Variant | Removed Component |
|---------|-------------------|
| Full ARTEMIS | None (baseline) |
| w/o Neural ODE | L1: Temporal ODE modeling |
| w/o Anomaly Memory | L2: Anomaly-aware memory |
| w/o Multi-hop | L3: Multi-hop broadcast |
| w/o Adv. Meta | L4: Adversarial meta-learning |
| w/o EWC | L5: Elastic weight consolidation |
| w/o Certified | L6: Certified training |

### Commands

```bash
# Full ablation study
python scripts/run_ablation_study.py \
    --config configs/default.yaml \
    --output results/ablation/

# Test specific variant
python scripts/run_ablation_study.py \
    --variant no_neural_ode \
    --output results/ablation_single/

# Quick mode
python scripts/run_ablation_study.py --quick
```

### Configuration Files

Ablation variants use configs in `configs/ablation/`:

```yaml
# configs/ablation/no_neural_ode.yaml
model:
  use_neural_ode: false
  use_anomaly_memory: true
  use_multi_hop: true
  # ... other settings remain true
```

### Expected Results

| Variant | Recall | Δ Recall | F1-Score | Δ F1 |
|---------|--------|----------|----------|------|
| Full ARTEMIS | 91.47% | - | 90.18% | - |
| w/o Neural ODE | 87.23% | -4.24% | 85.89% | -4.29% |
| w/o Anomaly Memory | 88.45% | -3.02% | 87.12% | -3.06% |
| w/o Multi-hop | 89.12% | -2.35% | 87.78% | -2.40% |
| w/o Adv. Meta | 89.78% | -1.69% | 88.45% | -1.73% |
| w/o EWC | 90.12% | -1.35% | 88.89% | -1.29% |
| w/o Certified | 90.89% | -0.58% | 89.56% | -0.62% |

---

## Experiment 3: Adversarial Robustness (Table 5)

### Description

Evaluates model robustness against adversarial attacks:
- **FGSM**: Fast Gradient Sign Method
- **PGD-10**: Projected Gradient Descent (10 steps)
- **PGD-20**: Projected Gradient Descent (20 steps)
- **Certified Accuracy**: Provable robustness at radius ε

### Commands

```bash
# Full adversarial evaluation
python scripts/run_adversarial_eval.py \
    --config configs/default.yaml \
    --output results/adversarial/

# Specific attack
python scripts/run_adversarial_eval.py \
    --attack pgd \
    --epsilon 0.1 \
    --steps 20

# Certified accuracy only
python scripts/run_adversarial_eval.py \
    --certified_only \
    --epsilon 0.05 0.1 0.15 0.2
```

### Attack Parameters

| Attack | Parameters |
|--------|------------|
| FGSM | ε ∈ {0.05, 0.1, 0.15, 0.2} |
| PGD-10 | ε ∈ {0.05, 0.1, 0.15, 0.2}, α=ε/4, steps=10 |
| PGD-20 | ε ∈ {0.05, 0.1, 0.15, 0.2}, α=ε/4, steps=20 |

### Expected Results (ε = 0.1)

| Method | Clean | FGSM | PGD-10 | PGD-20 | Certified |
|--------|-------|------|--------|--------|-----------|
| 2DynEthNet | 87.23% | 72.34% | 65.78% | 61.23% | - |
| ARTEMIS | 91.47% | 88.12% | 86.45% | 85.73% | 72.34% |

---

## Experiment 4: Efficiency Analysis (Table 6)

### Description

Measures computational efficiency:
- Training time per epoch
- Inference latency (ms/sample)
- GPU memory usage
- Model parameters

### Commands

```bash
# Full efficiency analysis
python scripts/run_efficiency_analysis.py \
    --config configs/default.yaml \
    --output results/efficiency/

# Batch size sweep
python scripts/run_efficiency_analysis.py \
    --batch_sizes 16 32 64 128 256

# Profile specific model
python scripts/run_efficiency_analysis.py \
    --model artemis \
    --profile
```

### Expected Results

| Method | Params (M) | Train (s/epoch) | Inference (ms) | Memory (GB) |
|--------|------------|-----------------|----------------|-------------|
| GraphSAGE | 0.42 | 12.3 | 2.1 | 2.8 |
| GAT | 0.58 | 15.7 | 2.8 | 3.2 |
| TGN | 1.23 | 28.4 | 5.2 | 5.4 |
| 2DynEthNet | 2.87 | 45.6 | 9.3 | 8.7 |
| **ARTEMIS** | 3.12 | 52.3 | 8.7 | 9.2 |

---

## Experiment 5: Continual Learning (Figure 6)

### Description

Evaluates catastrophic forgetting resistance through sequential task learning:
1. Train on Task 1, evaluate on Task 1
2. Train on Task 2, evaluate on Tasks 1-2
3. Continue until Task 6
4. Measure performance retention

### Commands

```bash
# Full continual learning evaluation
python scripts/run_continual_learning.py \
    --config configs/default.yaml \
    --tasks 1 2 3 4 5 6 \
    --output results/continual_learning/

# Custom task sequence
python scripts/run_continual_learning.py \
    --tasks 1 3 5 \
    --output results/continual_custom/

# Quick mode
python scripts/run_continual_learning.py --quick
```

### Expected Results

| Method | Avg. Forgetting | BWT | Final Acc |
|--------|-----------------|-----|-----------|
| TGN | 0.187 | -0.142 | 0.723 |
| 2DynEthNet | 0.156 | -0.118 | 0.756 |
| ARTEMIS (no EWC) | 0.134 | -0.098 | 0.789 |
| **ARTEMIS** | **0.067** | **-0.034** | **0.856** |

### Output Files

- `results/continual_learning/continual_learning_results.json`: Full results
- `results/continual_learning/figure6_forgetting_curves.png`: Visualization

---

## Statistical Analysis

All experiments include statistical significance testing:

### Methods Used

1. **Paired t-test**: For comparing ARTEMIS vs each baseline
2. **Wilcoxon signed-rank test**: Non-parametric alternative
3. **Cohen's d**: Effect size measurement
4. **Bootstrap confidence intervals**: 95% CI for metrics

### Significance Thresholds

- p < 0.001: Highly significant (***)
- p < 0.01: Very significant (**)
- p < 0.05: Significant (*)
- Cohen's d > 0.8: Large effect size

### Running Statistical Tests

```bash
# Included in main experiments
python scripts/run_main_experiments.py --config configs/default.yaml

# Results include:
# - p-values for all comparisons
# - Effect sizes
# - 95% confidence intervals
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
python scripts/run_main_experiments.py --config configs/default.yaml
# Edit config: training.batch_size: 16
```

**2. Slow Training**
```bash
# Enable mixed precision
# Edit config: training.mixed_precision: true

# Use multi-GPU
# Edit config: hardware.num_gpus: 4
```

**3. NaN Loss**
```bash
# Reduce learning rate
# Edit config: training.learning_rate: 0.0005

# Enable gradient clipping
# Edit config: training.gradient_clip: 0.5
```

**4. Dataset Not Found**
```bash
# Download dataset
python scripts/download_etgraph.py --output_dir ./data --task all

# Or use synthetic data
python scripts/download_etgraph.py --synthetic --num_graphs 1000
```

### Reproducing Exact Paper Results

To reproduce exact results from the paper:

1. Use the same random seeds: `--seed 42`
2. Use the provided config: `configs/default.yaml`
3. Run on recommended hardware: 4× RTX 3090
4. Use full ETGraph dataset (not synthetic)

### Contact

For issues or questions about reproducing results:
- Open an issue on GitHub
- Include: error message, config file, hardware specs

---

## Citation

If you use these experiments in your research, please cite:

```bibtex
@inproceedings{artemis2026,
  title={ARTEMIS: Adversarial-Resistant Temporal Embedding Model for Intelligent Security in Ethereum Phishing Detection},
  author={BlockchainLab},
  booktitle={ACM Conference on Computer and Communications Security (CCS)},
  year={2026}
}
```

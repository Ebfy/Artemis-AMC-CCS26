# ARTEMIS Hyperparameters Documentation

This document provides comprehensive documentation of all hyperparameters used in ARTEMIS, including their default values, recommended ranges, and tuning guidelines.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Innovation-Specific Parameters](#innovation-specific-parameters)
3. [Training Parameters](#training-parameters)
4. [Data Processing](#data-processing)
5. [Evaluation Settings](#evaluation-settings)
6. [Hyperparameter Tuning Guide](#hyperparameter-tuning-guide)

---

## Model Architecture

### Core Architecture Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hidden_dim` | 128 | [64, 256] | Hidden dimension for all layers |
| `num_layers` | 3 | [2, 4] | Number of GNN layers |
| `num_heads` | 4 | [2, 8] | Attention heads in GAT layers |
| `dropout` | 0.1 | [0.0, 0.3] | Dropout rate |
| `num_features` | 16 | Dataset-dependent | Input feature dimension |
| `num_classes` | 2 | Task-dependent | Output classes |

### Configuration Example

```yaml
model:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  num_features: 16
  num_classes: 2
```

### Layer-Specific Settings

```yaml
model:
  gat_layers:
    - {in: 16, out: 128, heads: 4, concat: true}
    - {in: 512, out: 128, heads: 4, concat: true}
    - {in: 512, out: 128, heads: 4, concat: false}
  
  pooling: "mean"  # Options: mean, max, add, attention
  
  classifier:
    - {in: 128, out: 64, activation: "relu"}
    - {in: 64, out: 2, activation: null}
```

---

## Innovation-Specific Parameters

### L1: Neural ODE Temporal Modeling

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_neural_ode` | true | bool | Enable Neural ODE |
| `ode_solver` | "dopri5" | ["dopri5", "rk4", "euler"] | ODE solver method |
| `ode_rtol` | 1e-3 | [1e-5, 1e-2] | Relative tolerance |
| `ode_atol` | 1e-4 | [1e-6, 1e-3] | Absolute tolerance |
| `stability_alpha` | 0.1 | [0.01, 0.5] | Lyapunov stability term |
| `time_hidden_dim` | 32 | [16, 64] | Time embedding dimension |

```yaml
neural_ode:
  enabled: true
  solver: "dopri5"
  rtol: 1e-3
  atol: 1e-4
  stability_alpha: 0.1
  time_hidden_dim: 32
  dynamics_layers: [128, 128, 128]
```

**Tuning Notes:**
- `dopri5` provides best accuracy but is slowest
- `rk4` is faster with slightly reduced accuracy
- `euler` is fastest but may be unstable for long time spans
- Increase `stability_alpha` if training becomes unstable

### L2: Anomaly-Aware Memory

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_anomaly_memory` | true | bool | Enable anomaly memory |
| `memory_size` | 1000 | [500, 5000] | Maximum memory entries |
| `memory_dim` | 128 | Match hidden_dim | Memory vector dimension |
| `anomaly_weight` | 0.5 | [0.1, 1.0] | Anomaly importance weight α |
| `mi_weight` | 0.5 | [0.1, 1.0] | Mutual information weight |
| `query_heads` | 4 | [2, 8] | Attention heads for queries |

```yaml
anomaly_memory:
  enabled: true
  memory_size: 1000
  memory_dim: 128
  anomaly_weight: 0.5
  mi_weight: 0.5
  query_heads: 4
  update_strategy: "importance"  # Options: importance, fifo, random
```

**Tuning Notes:**
- Larger `memory_size` captures more patterns but increases memory usage
- Higher `anomaly_weight` prioritizes rare patterns (good for imbalanced data)
- Use `update_strategy: "fifo"` for faster training (but lower quality)

### L3: Multi-Hop Broadcast

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_multi_hop` | true | bool | Enable multi-hop broadcast |
| `num_hops` | 3 | [2, 5] | Maximum broadcast hops |
| `hop_attention` | true | bool | Use attention for hop weighting |
| `hop_decay` | 0.8 | [0.5, 1.0] | Decay factor per hop |

```yaml
multi_hop:
  enabled: true
  num_hops: 3
  hop_attention: true
  hop_decay: 0.8
  aggregation: "attention"  # Options: attention, mean, max
```

**Tuning Notes:**
- More hops capture longer-range dependencies but increase computation
- `hop_decay` prevents over-smoothing from distant nodes
- Use fewer hops (2) for dense graphs, more (4-5) for sparse graphs

### L4: Adversarial Meta-Learning

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_adversarial_meta` | true | bool | Enable adversarial meta-learning |
| `inner_lr` | 0.01 | [0.001, 0.1] | Inner loop learning rate |
| `outer_lr` | 0.001 | [0.0001, 0.01] | Outer loop learning rate |
| `inner_steps` | 5 | [3, 10] | Inner loop optimization steps |
| `adversarial_ratio` | 0.3 | [0.1, 0.5] | Ratio of adversarial tasks |
| `meta_batch_size` | 4 | [2, 8] | Tasks per meta-batch |

```yaml
adversarial_meta:
  enabled: true
  inner_lr: 0.01
  outer_lr: 0.001
  inner_steps: 5
  adversarial_ratio: 0.3
  meta_batch_size: 4
  pgd_steps: 10
  pgd_epsilon: 0.1
```

**Tuning Notes:**
- Higher `adversarial_ratio` improves robustness but may hurt clean accuracy
- Increase `inner_steps` for better adaptation (but slower training)
- `meta_batch_size` limited by GPU memory

### L5: Elastic Weight Consolidation

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_ewc` | true | bool | Enable EWC |
| `ewc_lambda` | 1000 | [100, 10000] | EWC regularization strength |
| `fisher_samples` | 1000 | [500, 5000] | Samples for Fisher estimation |
| `online_ewc` | true | bool | Use online Fisher updates |
| `ewc_decay` | 0.9 | [0.5, 0.99] | Fisher decay for online EWC |

```yaml
ewc:
  enabled: true
  lambda: 1000
  fisher_samples: 1000
  online: true
  decay: 0.9
```

**Tuning Notes:**
- Higher `ewc_lambda` reduces forgetting but may limit new learning
- Use `online_ewc: true` for streaming scenarios
- Increase `fisher_samples` for more accurate importance estimates

### L6: Certified Adversarial Training

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_certified` | true | bool | Enable certified training |
| `cert_epsilon` | 0.1 | [0.05, 0.2] | Certification radius |
| `smoothing_sigma` | 0.25 | [0.1, 0.5] | Randomized smoothing σ |
| `num_smoothing_samples` | 100 | [50, 500] | Samples for certification |
| `pgd_train_steps` | 10 | [5, 20] | PGD steps during training |
| `pgd_train_epsilon` | 0.1 | [0.05, 0.2] | PGD ε during training |

```yaml
certified:
  enabled: true
  epsilon: 0.1
  smoothing_sigma: 0.25
  num_samples: 100
  confidence: 0.001
  pgd_steps: 10
  pgd_epsilon: 0.1
  pgd_alpha: 0.025
```

**Tuning Notes:**
- Larger `smoothing_sigma` gives larger certified radius but lower clean accuracy
- More `num_smoothing_samples` gives tighter bounds but slower certification
- Trade-off: certified accuracy vs clean accuracy

---

## Training Parameters

### Optimizer Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `optimizer` | "adamw" | ["adam", "adamw", "sgd"] | Optimizer type |
| `learning_rate` | 0.001 | [0.0001, 0.01] | Initial learning rate |
| `weight_decay` | 0.01 | [0.0, 0.1] | L2 regularization |
| `beta1` | 0.9 | [0.8, 0.99] | Adam β1 |
| `beta2` | 0.999 | [0.99, 0.9999] | Adam β2 |

```yaml
training:
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]
```

### Learning Rate Schedule

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `scheduler` | "cosine" | ["cosine", "step", "plateau"] | LR scheduler |
| `warmup_epochs` | 5 | [0, 20] | Warmup period |
| `min_lr` | 1e-6 | [1e-7, 1e-5] | Minimum learning rate |
| `T_max` | 100 | Epochs | Cosine annealing period |

```yaml
training:
  scheduler: "cosine"
  warmup_epochs: 5
  min_lr: 1e-6
  T_max: 100
```

### Training Control

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epochs` | 100 | [50, 200] | Maximum training epochs |
| `batch_size` | 32 | [16, 128] | Training batch size |
| `gradient_clip` | 1.0 | [0.5, 5.0] | Gradient clipping norm |
| `early_stopping_patience` | 15 | [5, 30] | Early stopping patience |
| `mixed_precision` | true | bool | Use FP16 training |

```yaml
training:
  epochs: 100
  batch_size: 32
  gradient_clip: 1.0
  early_stopping_patience: 15
  mixed_precision: true
  accumulation_steps: 1
```

---

## Data Processing

### Temporal Windowing

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `window_size_hours` | 2 | [1, 24] | Time window size |
| `stride_minutes` | 30 | [15, 120] | Window stride |
| `min_nodes` | 10 | [5, 50] | Minimum nodes per window |
| `min_edges` | 20 | [10, 100] | Minimum edges per window |

```yaml
data:
  window_size_hours: 2
  stride_minutes: 30
  min_nodes: 10
  min_edges: 20
```

### Data Augmentation

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `node_dropout` | 0.1 | [0.0, 0.3] | Node dropout rate |
| `edge_dropout` | 0.1 | [0.0, 0.3] | Edge dropout rate |
| `feature_noise` | 0.01 | [0.0, 0.1] | Feature noise std |

```yaml
data:
  augmentation:
    node_dropout: 0.1
    edge_dropout: 0.1
    feature_noise: 0.01
    time_jitter: 0.01
```

### Train/Val/Test Split

```yaml
data:
  split:
    train: 0.70
    val: 0.15
    test: 0.15
  stratify: true
  temporal_split: true  # Respect time ordering
```

---

## Evaluation Settings

### Metrics

```yaml
evaluation:
  primary_metric: "recall"  # For early stopping
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - auc_roc
    - fpr
    - specificity
    - mcc
```

### Statistical Testing

```yaml
evaluation:
  statistical_tests:
    enabled: true
    significance_level: 0.01
    tests:
      - paired_ttest
      - wilcoxon
    effect_size: "cohens_d"
    bootstrap_samples: 1000
    confidence_level: 0.95
```

### Adversarial Evaluation

```yaml
evaluation:
  adversarial:
    attacks:
      - {name: "fgsm", epsilon: [0.05, 0.1, 0.15, 0.2]}
      - {name: "pgd", epsilon: [0.05, 0.1, 0.15, 0.2], steps: 10}
      - {name: "pgd", epsilon: [0.05, 0.1, 0.15, 0.2], steps: 20}
    certified:
      epsilon: [0.05, 0.1, 0.15, 0.2]
      sigma: 0.25
      n_samples: 100
```

---

## Hyperparameter Tuning Guide

### Recommended Tuning Order

1. **Learning rate**: Most impactful, tune first
2. **Hidden dimension**: Balance capacity and efficiency
3. **Batch size**: Affects convergence and memory
4. **Dropout**: Regularization strength
5. **Innovation-specific**: Fine-tune each component

### Tuning for Different Scenarios

#### High Recall Priority (Fraud Detection)
```yaml
model:
  hidden_dim: 128
  dropout: 0.05  # Less regularization
training:
  learning_rate: 0.0005
  weight_decay: 0.005
evaluation:
  primary_metric: "recall"
```

#### Balanced Performance
```yaml
model:
  hidden_dim: 128
  dropout: 0.1
training:
  learning_rate: 0.001
  weight_decay: 0.01
evaluation:
  primary_metric: "f1_score"
```

#### Maximum Robustness
```yaml
adversarial_meta:
  adversarial_ratio: 0.5
certified:
  smoothing_sigma: 0.3
  pgd_steps: 20
training:
  learning_rate: 0.0003
```

#### Low Memory (Single GPU)
```yaml
model:
  hidden_dim: 64
  num_heads: 2
anomaly_memory:
  memory_size: 500
training:
  batch_size: 16
  accumulation_steps: 4
  mixed_precision: true
```

### Grid Search Configurations

```python
# Recommended grid search space
param_grid = {
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
    "hidden_dim": [64, 128, 256],
    "dropout": [0.0, 0.1, 0.2],
    "num_heads": [2, 4, 8],
    "ewc_lambda": [100, 1000, 10000],
}
```

### Bayesian Optimization Bounds

```python
# For Optuna/Ray Tune
bounds = {
    "learning_rate": (1e-5, 1e-2, "log"),
    "hidden_dim": (32, 256, "int"),
    "dropout": (0.0, 0.3, "float"),
    "ewc_lambda": (10, 10000, "log"),
    "smoothing_sigma": (0.1, 0.5, "float"),
}
```

---

## Default Configuration File

The complete default configuration is in `configs/default.yaml`:

```yaml
# Full default configuration
model:
  hidden_dim: 128
  num_layers: 3
  num_heads: 4
  dropout: 0.1
  num_features: 16
  num_classes: 2
  
  # Innovations
  use_neural_ode: true
  use_anomaly_memory: true
  use_multi_hop: true
  use_adversarial_meta: true
  use_ewc: true
  use_certified: true

neural_ode:
  solver: "dopri5"
  rtol: 1e-3
  atol: 1e-4
  stability_alpha: 0.1

anomaly_memory:
  memory_size: 1000
  anomaly_weight: 0.5

multi_hop:
  num_hops: 3
  hop_attention: true

adversarial_meta:
  inner_lr: 0.01
  outer_lr: 0.001
  adversarial_ratio: 0.3

ewc:
  lambda: 1000
  online: true

certified:
  epsilon: 0.1
  smoothing_sigma: 0.25

training:
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.01
  epochs: 100
  batch_size: 32
  gradient_clip: 1.0
  early_stopping_patience: 15
  mixed_precision: true

data:
  window_size_hours: 2
  stride_minutes: 30
  split:
    train: 0.70
    val: 0.15
    test: 0.15

evaluation:
  primary_metric: "recall"
```

---

## Citation

```bibtex
@inproceedings{artemis2026,
  title={ARTEMIS: Adversarial-Resistant Temporal Embedding Model for Intelligent Security},
  author={BlockchainLab},
  booktitle={ACM CCS},
  year={2026}
}
```

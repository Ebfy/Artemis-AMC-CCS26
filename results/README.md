# Results Directory

Experiment results will be saved here.

## Expected Output Structure

After running experiments, results will be organized as:

```
results/
├── main_performance/
│   ├── metrics.json
│   ├── summary.txt
│   └── table3.csv
├── ablation/
│   ├── ablation_results.json
│   └── table4.csv
├── adversarial/
│   ├── adversarial_results.json
│   ├── table5.csv
│   └── robustness_curves.png
├── efficiency/
│   ├── efficiency_results.json
│   └── table6.csv
└── continual_learning/
    ├── continual_learning_results.json
    └── figure6_forgetting_curves.png
```

## Running Experiments

```bash
# Main performance (Table 3)
python scripts/run_main_experiments.py --output results/main_performance/

# Ablation study (Table 4)
python scripts/run_ablation_study.py --output results/ablation/

# Adversarial evaluation (Table 5)
python scripts/run_adversarial_eval.py --output results/adversarial/

# Efficiency analysis (Table 6)
python scripts/run_efficiency_analysis.py --output results/efficiency/

# Continual learning (Figure 6)
python scripts/run_continual_learning.py --output results/continual_learning/
```

## Result Files

- **metrics.json**: Raw metrics for all models and tasks
- **summary.txt**: Human-readable summary
- **table*.csv**: LaTeX-ready tables
- **figure*.png**: Visualizations

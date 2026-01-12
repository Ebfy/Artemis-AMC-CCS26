# Data Directory

This directory should contain the ETGraph dataset.

## Download Instructions

```bash
# Download full dataset (~15GB)
python scripts/download_etgraph.py --output_dir ./data --task all

# Or create synthetic data for testing
python scripts/download_etgraph.py --output_dir ./data --synthetic --num_graphs 1000
```

## Expected Structure

After download, the directory should contain:

```
data/
├── task1/
│   ├── nodes.csv
│   ├── edges.csv
│   ├── labels.csv
│   └── timestamps.csv
├── task2/
│   └── ...
├── task3/
│   └── ...
├── task4/
│   └── ...
├── task5/
│   └── ...
├── task6/
│   └── ...
└── processed/
    └── (preprocessed graph data)
```

## Dataset Information

- **Source**: ETGraph (Ethereum Transaction Graph)
- **Size**: ~15 GB
- **Tasks**: 6 phishing detection tasks
- **Total Nodes**: ~3M
- **Total Edges**: ~13.5M
- **Phishing Labels**: ~1,165 addresses

## Citation

If you use this dataset, please cite:

```bibtex
@article{etgraph2023,
  title={2DynEthNet: A Two-Stage Dynamic Graph Neural Network for Ethereum Phishing Detection},
  author={...},
  journal={...},
  year={2023}
}
```

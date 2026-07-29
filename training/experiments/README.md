# Experiment Configurations and Running Registry

This directory contains configuration templates managing reproducible training runs.

## Folder Structure

```
experiments/
├── exp01_baseline.yaml            # Control group (Train Student from scratch)
├── exp02_distillation.yaml        # Main group (Train Student with Teacher KD)
├── exp03_residual_scaling.yaml    # Ablation (Study Student capacity scaling)
├── README.md                      # This documentation index
└── results/                       # Outputs folder storing running logs and results
```

## Running an Experiment

To launch a training run using any configuration template:
```powershell
python train.py --config experiments/exp01_baseline.yaml
```

All checkpoints, console log metrics, and tensorboard logs will be organized under `outputs/runs/<experiment_name>/`.
Result summary statistics are committed to `experiments/results/`.

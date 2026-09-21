# RoboDojo Data and Training Guide

[中文](robodojo_zh.md) | [Data index](README.md) | [Back to project README](../../../README.md)

This guide covers RoboDojo training-data preparation and training in EasyWAM.

## Training Data

Download the ready-to-use LeRobot v3.0 dataset from [OpenMOSS-Team/robodojo-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/robodojo-lerobot-v3.0). The destination matches the default path in `configs/data/robodojo.yaml`:

```bash
huggingface-cli download OpenMOSS-Team/robodojo-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/robodojo-lerobot-v3.0
```

The default `configs/data/robodojo.yaml` expects:

```text
data/robodojo-lerobot-v3.0/
├── data/
├── dataset_stats.json
├── meta/
└── videos/
```

The data config reads `cam_high`, `cam_left_wrist`, and `cam_right_wrist`, plus 14-D joint state/action. It forms a 384×320 three-camera mosaic at 9 video timestamps per 32-step action horizon and applies global z-score normalization. A fixed-seed episode split reserves 1% for validation.

## Training

Precompute the RoboDojo Wan2.2 text cache. Each prompt is stored in a SHA-256-named file and shared by the five task variants:

```bash
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22
```

The five Wan2.2 recipes are `mot`, `hidden`, `unified`, `mot_joint`, and `mot_idm`, named `robodojo_easywam_<architecture>_wan22`. Their training values match the RoboTwin recipes. Launch training by passing the task as a Hydra override:

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=robodojo_easywam_mot_wan22
```

Use another `robodojo_*.yaml` recipe in `configs/task/` for a different architecture. The default data config loads normalization statistics from `data/robodojo-lerobot-v3.0/dataset_stats.json`. Training outputs, including the matching statistics, are saved under `runs/<task>/<run-id>/`.

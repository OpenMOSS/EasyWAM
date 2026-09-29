# RoboTwin Data and Training Guide

[中文](robotwin_zh.md) | [Data index](README.md) | [Evaluation guide](../benchmark/robotwin.md) | [Back to project README](../../../README.md)

This guide covers RoboTwin 2.0 training-data preparation and training in EasyWAM.

## Training Data

Obtain the LeRobot v3.0 data from [OpenMOSS-Team/robotwin2.0-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/robotwin2.0-lerobot-v3.0), and place the clean and randomized datasets in the child directories expected by `configs/data/robotwin.yaml`:

```bash
huggingface-cli download OpenMOSS-Team/robotwin2.0-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/robotwin2.0-lerobot-v3.0
```

The default `configs/data/robotwin.yaml` expects:

```text
data/robotwin2.0-lerobot-v3.0/
├── clean/                 # 50 tasks × 50 trajectories
├── random/                # 50 tasks × 500 trajectories
└── dataset_stats.json     # normalization statistics for full mode
```

EasyWAM requires LeRobot v3.0 datasets for training.

The pipeline combines the high camera and two wrist cameras into a 384×320 video. It retains all 33 action/state steps and decodes 9 sparse video timestamps.

## Training

Precompute the RoboTwin text cache. Each prompt is written to its own SHA-256-named file and loaded on demand during training:

```bash
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_wan22
```

Select one of the five Wan2.2 task recipes:

| Architecture | Task |
| --- | --- |
| MoT | `robotwin_easywam_mot_wan22` |
| Hidden | `robotwin_easywam_hidden_wan22` |
| Unified | `robotwin_easywam_unified_wan22` |
| MoT-Joint | `robotwin_easywam_mot_joint_wan22` |
| MoT-IDM | `robotwin_easywam_mot_idm_wan22` |

For example:

```bash
# Full training (default): clean + random at the original 1:10 ratio
NPROC_PER_NODE=8 bash scripts/train_zero2.sh \
  task=robotwin_easywam_mot_wan22 data.mode=full

# Clean-only / random-only
NPROC_PER_NODE=8 bash scripts/train_zero2.sh \
  task=robotwin_easywam_mot_wan22 data.mode=clean
NPROC_PER_NODE=8 bash scripts/train_zero2.sh \
  task=robotwin_easywam_mot_wan22 data.mode=random
```

`data.mode` selects the same directories for training, validation, and text-embedding precomputation. Clean and random use the `dataset_stats.json` in their respective child directory; full uses the aggregate file in the parent directory. Set `data.train.pretrained_norm_stats=null` and `data.val.pretrained_norm_stats=null` if the statistics must be recomputed for a different dataset.

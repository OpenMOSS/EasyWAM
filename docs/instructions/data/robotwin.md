# RoboTwin Data and Training Guide

[中文](robotwin_zh.md) | [Data index](README.md) | [Evaluation guide](../benchmark/robotwin.md) | [Back to project README](../../../README.md)

This guide covers RoboTwin 2.0 training-data preparation and training in EasyWAM.

## Training Data

Download the ready-to-use LeRobot v3.0 dataset from [OpenMOSS-Team/robotwin2.0-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/robotwin2.0-lerobot-v3.0). The destination matches the default path in `configs/data/robotwin.yaml`:

```bash
huggingface-cli download OpenMOSS-Team/robotwin2.0-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/robotwin2.0-lerobot-v3.0
```

The default `configs/data/robotwin.yaml` expects:

```text
data/robotwin2.0-lerobot-v3.0/
├── data/
├── dataset_stats.json
├── meta/
└── videos/
```

EasyWAM remains compatible with LeRobot v2.1 datasets and provides `scripts/convert_lerobot_v21_to_v30.py` for users who want to convert them to v3.0.

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
NPROC_PER_NODE=8 bash scripts/train_zero2.sh task=robotwin_easywam_mot_wan22
```

The default data config loads normalization statistics from `data/robotwin2.0-lerobot-v3.0/dataset_stats.json`. Set `data.train.pretrained_norm_stats=null` and `data.val.pretrained_norm_stats=null` if the statistics must be recomputed for a different dataset.

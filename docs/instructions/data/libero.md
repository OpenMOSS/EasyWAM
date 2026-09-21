# LIBERO Data and Training Guide

[中文](libero_zh.md) | [Data index](README.md) | [Evaluation guide](../benchmark/libero.md) | [Back to project README](../../../README.md)

This guide covers LIBERO training-data preparation and training in EasyWAM.

## Training Data

Download the ready-to-use LeRobot v3.0 dataset from [OpenMOSS-Team/libero-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/libero-lerobot-v3.0). The destination matches the default path in `configs/data/libero.yaml`:

```bash
huggingface-cli download OpenMOSS-Team/libero-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/libero-lerobot-v3.0
```

The default `configs/data/libero.yaml` expects:

```text
data/libero-lerobot-v3.0/
├── libero_10/
├── libero_goal/
├── libero_object/
└── libero_spatial/
```

EasyWAM remains compatible with LeRobot v2.1 datasets and provides `scripts/convert_lerobot_v21_to_v30.py` for users who want to convert them to v3.0.

The pipeline concatenates the agent and wrist cameras at 224 px resolution. It retains all 33 action/state steps while decoding only the 9 video timestamps `[0, 4, ..., 32]`.

## Training

Precompute the per-instruction text embedding cache once. Each prompt is stored in a file named by its SHA-256 hash, and models using the same text encoder can share the cache:

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
python scripts/precompute_text_embeds.py task=libero_easywam_mot_cosmos25
```

Multi-GPU cache generation is also supported:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
```

Select one of the current task names:

| Model | Wan full | Wan LoRA | Cosmos full | Cosmos LoRA |
| --- | --- | --- | --- | --- |
| EasyWAM-MoT | `libero_easywam_mot_wan22` | `libero_easywam_mot_wan22_lora` | `libero_easywam_mot_cosmos25` | `libero_easywam_mot_cosmos25_lora` |
| EasyWAM-Unified | `libero_easywam_unified_wan22` | `libero_easywam_unified_wan22_lora` | `libero_easywam_unified_cosmos25` | `libero_easywam_unified_cosmos25_lora` |
| EasyWAM-Hidden | `libero_easywam_hidden_wan22` | `libero_easywam_hidden_wan22_lora` | `libero_easywam_hidden_cosmos25` | `libero_easywam_hidden_cosmos25_lora` |

Launch training by passing the task as a Hydra override:

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot_wan22

NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_cosmos25
```

Use `scripts/train_zero2.sh` or `scripts/train_zero2_offload.sh` for ZeRO-2 or ZeRO-2 CPU offload. The run is written to `runs/<task>/<run-id>/`. If no pretrained normalization statistics are configured, the first run computes and saves `dataset_stats.json` in the run directory; use the matching file for evaluation.

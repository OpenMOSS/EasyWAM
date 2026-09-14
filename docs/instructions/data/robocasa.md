# RoboCasa365 Data and Training Guide

[中文](robocasa_zh.md) | [Data index](README.md) | [Evaluation guide](../benchmark/robocasa.md) | [Back to project README](../../../README.md)

This guide covers RoboCasa365 training-data preparation and training in EasyWAM.

## Download

The EMBER mirrors are already flat LeRobot v3.0 datasets, so no conversion is needed:

```bash
huggingface-cli download ember-lab-berkeley/robocasa365-pretrain-atomic \
  --repo-type dataset \
  --local-dir data/robocasa365-lerobot-v3.0/pretrain-atomic

huggingface-cli download ember-lab-berkeley/robocasa365-pretrain-composite \
  --repo-type dataset \
  --local-dir data/robocasa365-lerobot-v3.0/pretrain-composite
```

The checked-in **configs/data/robocasa.yaml** expects:

```text
data/robocasa365-lerobot-v3.0/
├── pretrain-atomic/
└── pretrain-composite/
```

**pretrain** is the broad foundation-training corpus. **target** contains benchmark-task demonstrations intended for post-training or fine-tuning; it is not included in this initial recipe.

The adapter preserves the dataset schema exactly: three 256 px cameras are concatenated left, right, then wrist; state is base_pos(3) + base_quat(4) + eef_pos_rel(3) + eef_quat_rel(4) + gripper_qpos(2); action is base_motion(4) + control_mode(1) + eef_delta_pos(3) + eef_delta_axis_angle(3) + gripper(1).

## Statistics and training

Precompute text embeddings and start training:

```bash
python scripts/precompute_text_embeds.py task=robocasa_easywam_mot_wan22
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=robocasa_easywam_mot_wan22
```

Use the matching task recipe for another model or LoRA training. No pretrained normalization file is configured. During the first run, EasyWAM computes the combined atomic-and-composite statistics and writes them to:

```text
runs/robocasa_easywam_mot_wan22/<run-id>/dataset_stats.json
```

Keep this file with the checkpoint and pass it to evaluation. The empty upstream **meta/stats.json** files are not model normalization statistics.

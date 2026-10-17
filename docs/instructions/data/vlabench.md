# VLABench data and training

[中文](vlabench_zh.md) | [Data index](README.md) | [Evaluation](../benchmark/vlabench.md)

Run these commands from the project root.

## Training data

Download the ready-to-use LeRobot v3.0 dataset from [OpenMOSS-Team/vlabench-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/vlabench-lerobot-v3.0). No conversion is required:

```bash
hf download OpenMOSS-Team/vlabench-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/vlabench-lerobot-v3.0
```

This location matches `configs/data/vlabench.yaml`:

```text
data/vlabench-lerobot-v3.0/
├── data/
├── dataset_stats.json
├── meta/
├── README.md
└── videos/
```

The dataset contains 5000 episodes and 575101 frames at 10 FPS. Cameras `image`, `wrist_image`, and `second_image` form a 384×320 observation. States and actions use 10-dimensional `[xyz, rotation6d, gripper]`, with positions and orientations in the robot base frame and rotation6d storing the first two rotation-matrix rows. Gripper value 1 means closed in state and open in action.

The default configuration uses 32 actions and 9 video frames, loads global min/max normalization statistics from `dataset_stats.json`, and does not hold out a separate validation set. Training reuses the existing LeRobot loader and WAMProcessor and needs no additional training dependencies.

## Training

Prepare Wan2.2 weights as described in the project README, then precompute text embeddings:

```bash
python scripts/precompute_text_embeds.py task=vlabench_easywam_mot_wan22
```

Five Wan2.2 recipes share this cache: `mot`, `hidden`, `unified`, `mot_joint`, and `mot_idm`, named `vlabench_easywam_<architecture>_wan22` in `configs/task/`. Use the same precision for caching and training.

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=vlabench_easywam_mot_wan22
```

Adjust `NPROC_PER_NODE` to your GPU count. Defaults are batch 4 per process, BF16, learning rate 1e-4, 30000 steps, and checkpoints every 1000 steps. Training outputs and accompanying statistics are saved under `runs/<task>/<run-id>/`. Evaluate with the checkpoint and normalization statistics matching that training run; see the [evaluation guide](../benchmark/vlabench.md).

# LIBERO Guide

[中文](libero_zh.md) | [Back to README](../README.md)

This guide covers LIBERO data preparation, training, and evaluation in EasyWAM.

## Environment and Data

Install EasyWAM first, then install the official [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) package in the same environment. The preprocessed data used by the current configs was generated with MuJoCo 3.3.2, so keep the simulator version aligned:

```bash
pip install mujoco==3.3.2
```

Download the four archives from [LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam), then extract them:

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2
for archive in *.tar.gz; do
  tar -xzf "$archive"
done
cd ../..
```

The default `configs/data/libero_2cam.yaml` expects:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

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

## Evaluation

Install the official LIBERO simulator before launching the manager. Evaluate a trained checkpoint with:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=./runs/libero_easywam_mot_wan22/<run-id>/checkpoints/weights/<checkpoint>.pt \
  EVALUATION.dataset_stats_path=./runs/libero_easywam_mot_wan22/<run-id>/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

Useful overrides:

```bash
# Evaluate selected suites with one worker on each of four GPUs
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object]' \
  MULTIRUN.num_gpus=4 MULTIRUN.max_tasks_per_gpu=1

# Validate installation and create the task manifest without starting rollouts
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> MULTIRUN.create_only=true
```

The default protocol evaluates all four suites for 50 trials per task. With one worker per GPU the manager selects EGL; multiple workers per GPU use OSMesa. Videos and progress rendering are disabled by default and can be controlled with `EVALUATION.video_mode`, `EVALUATION.visualize_future_video`, and `EVALUATION.progress`.

Results are stored under `evaluate_results/libero/<task>/<timestamp>/`, including worker logs, task JSON files, `summary.json`, `summary.csv`, and `task_success_rates.csv`. Reusing an explicit `EVALUATION.output_dir` resumes the run: valid completed task results are skipped.

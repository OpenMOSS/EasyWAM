# LIBERO Evaluation Guide

[中文](libero_zh.md) | [Benchmark index](README.md) | [Data and training guide](../data/libero.md) | [Back to project README](../../../README.md)

Install EasyWAM and the official [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulator in the same environment. The project uses MuJoCo 3.3.2:

```bash
pip install mujoco==3.3.2
```

Evaluate a trained checkpoint with its matching normalization statistics:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=./runs/libero_easywam_mot_wan22/<run-id>/checkpoints/weights/<checkpoint>.pt \
  EVALUATION.dataset_stats_path=./runs/libero_easywam_mot_wan22/<run-id>/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

Useful overrides:

```bash
# Evaluate selected suites with four environments per GPU
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object]' \
  MULTIRUN.num_gpus=4 MULTIRUN.env_num_per_gpu=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10

# Validate installation and create the task manifest without starting rollouts
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> MULTIRUN.create_only=true
```

The default protocol evaluates all four suites for 50 trials per task. Each GPU loads one model; environments claim tasks dynamically and share its inference batcher. One environment uses EGL and concurrent environments use OSMesa. Videos and progress rendering are disabled by default and can be controlled with `EVALUATION.video_mode`, `EVALUATION.visualize_future_video`, and `EVALUATION.progress`.

Results are stored under `evaluate_results/libero/<task>/<timestamp>/`, including worker logs, task JSON files, `summary.json`, `summary.csv`, and `task_success_rates.csv`. Reusing an explicit `EVALUATION.output_dir` resumes the run by skipping valid completed tasks.

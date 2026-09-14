# LIBERO Evaluation Guide

[中文](libero_zh.md) | [Benchmark index](README.md) | [Data and training guide](../data/libero.md) | [Back to project README](../../../README.md)

## Installation

Install LIBERO in the same Python 3.10 environment as EasyWAM:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO

# Only packages needed by LIBERO evaluation that are not declared by EasyWAM.
pip install mujoco==3.3.2 easydict==1.9 \
  robosuite==1.4.0 bddl==1.0.1 future==0.18.2 \
  cloudpickle==2.1.0 gym==0.25.2

# LIBERO's setup.py has no runtime dependencies; avoid reinstalling EasyWAM packages.
pip install --no-deps -e third_party/LIBERO
```

On the first import, LIBERO asks where to store demonstration datasets. Evaluation only uses the BDDL files, assets, and initial states bundled with the source tree, so accept the default:

```bash
printf 'n\n' | python -c "import libero.libero"
```

This creates `~/.libero/config.yaml`. If `LIBERO_CONFIG_PATH` is set, it is created under that directory instead; keep that environment variable set when running evaluation. The official requirements also list `robomimic` and `thop`, but EasyWAM does not use LIBERO's built-in policy training stack, so they are not needed here. EasyWAM already provides Hydra, NumPy, Weights & Biases, Transformers, Einops, PyTorch, Pillow, Termcolor, and tqdm.

See the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) for the upstream source and dependency versions.

## Evaluation

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
# Evaluate selected suites with two model workers per GPU and four actors per worker
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object]' \
  MULTIRUN.num_gpus=3 'MULTIRUN.gpu_ids=[1,3,5,7]' \
  MULTIRUN.workers_per_gpu=2 \
  MULTIRUN.env_num_per_worker=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10
```

The default protocol evaluates all four suites for 50 trials per task. With `MULTIRUN.gpu_ids=null`, the first `num_gpus` devices are used; otherwise the first `num_gpus` entries of the ordered `gpu_ids` candidate pool are selected. Workers are distributed across the selected GPUs in round-robin order; every worker loads its own model, and its rollout actors dynamically claim tasks and share that worker's inference batcher. Increasing `workers_per_gpu` increases model memory use. One active environment uses EGL and concurrent environments on the same GPU use OSMesa. Videos and progress rendering are disabled by default and can be controlled with `EVALUATION.video_mode`, `EVALUATION.visualize_future_video`, and `EVALUATION.progress`.

Results are stored under `evaluate_results/libero/<task>/<timestamp>/`, including worker logs, task JSON files, `summary.json`, `summary.csv`, and `task_success_rates.csv`. Reusing an explicit `EVALUATION.output_dir` resumes the run by skipping valid completed tasks.

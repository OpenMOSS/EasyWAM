# RoboCasa365 Evaluation Guide

[中文](robocasa_zh.md) | [Benchmark index](README.md) | [Data and training guide](../data/robocasa.md) | [Back to project README](../../../README.md)

The evaluator implements the official Human300 multitask protocol: atomic_seen (18 tasks), composite_seen (16), and composite_unseen (16), with 50 episodes per task and the task-specific horizon from RoboCasa v1.0.1 or newer.

## Install in the EasyWAM environment

Install robosuite master and the official RoboCasa checkout in the same Python environment as EasyWAM. robosuite is available on PyPI, but RoboCasa explicitly requires the newer master branch; pip can install that branch directly:

```bash
pip install "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@master"

git clone https://github.com/robocasa/robocasa.git third_party/robocasa
pip install gymnasium pygame h5py lxml hidapi "tianshou==0.4.10"
pip install --no-deps -e third_party/robocasa

python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

The no-dependencies install is intentional: RoboCasa's package pins NumPy and LeRobot versions that conflict with EasyWAM's training environment. EasyWAM already includes imageio, imageio-ffmpeg, termcolor, Pillow, tqdm, NumPy, and PyYAML; robosuite installs OpenCV, pynput, SciPy, Numba, and its other base dependencies. The command above therefore lists only the remaining RoboCasa runtime packages. Keep EasyWAM's existing NumPy, MuJoCo, datasets, and LeRobot-compatible stack. Because both simulators share robosuite/MuJoCo dependencies, changing this environment can affect an existing LIBERO installation.

## Evaluate

Run all 50 tasks on the pretrain scene/object split:

```bash
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 \
  ckpt=./runs/robocasa_easywam_mot_wan22/<run-id>/checkpoints/weights/<checkpoint>.pt \
  EVALUATION.dataset_stats_path=./runs/robocasa_easywam_mot_wan22/<run-id>/dataset_stats.json
```

The default is 8 GPUs, two model workers per GPU, 20 rollout actors per worker, inference batches up to 16, a 5 ms batching window, a 32-action prediction horizon, and replanning every 16 actions. These defaults are resource-intensive; reduce `workers_per_gpu`, `env_num_per_worker`, and `inference_batch_size` together when memory is limited. With `MULTIRUN.gpu_ids=null`, the first `num_gpus` devices are used; otherwise the first `num_gpus` entries of the ordered `gpu_ids` candidate pool are selected. Workers are assigned to the selected GPUs in round-robin order. Every worker loads its own model and owns an independent dynamic inference batcher, so increasing `workers_per_gpu` also increases GPU memory use.

Useful overrides:

```bash
# Evaluate one official group on four GPUs
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 ckpt=<checkpoint> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  'MULTIRUN.task_sets=[atomic_seen]' MULTIRUN.num_gpus=4 \
  'MULTIRUN.gpu_ids=[1,3,5,7]' \
  MULTIRUN.workers_per_gpu=2 MULTIRUN.env_num_per_worker=4 \
  MULTIRUN.inference_batch_size=4

# Evaluate one task
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 ckpt=<checkpoint> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  EVALUATION.task_name=CloseFridge

# Evaluate target scenes and objects
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 ckpt=<checkpoint> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  EVALUATION.split=target
```

Set **EVALUATION.video_mode=failures** or **all** to save three-camera rollout videos. Videos are off by default.

Results are stored under **evaluate_results/robocasa/robocasa_easywam_mot_wan22/<timestamp>/**. Each task has a result JSON and optional videos; the run root contains worker logs, the resolved config, simulator version/commit metadata, **summary.json**, **summary.csv**, and **task_success_rates.csv**. A worker failure still writes a partial summary before returning an error. The overall rate is weighted across all evaluated episodes (equivalent to the 50-task leaderboard mean when every task has 50 trials).

To resume, reuse the exact output directory:

```bash
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 ckpt=<checkpoint> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  EVALUATION.output_dir=<existing-output-directory>
```

Only results with the matching task set, task name, split, episode count, and a complete success/failure episode partition are skipped.

# VLABench evaluation

[中文](vlabench_zh.md) | [Benchmark index](README.md) | [Data and training](../data/vlabench.md)

The entrypoint is `experiments/vlabench/run_vlabench_manager.py`, configured by `configs/benchmark/sim_vlabench.yaml`. Run all commands from the project root.

## Installation

Install the following additional dependencies in the current environment. NumPy, h5py, OpenCV, imageio, PyYAML, and msgpack are already provided by the project and its dependencies and are omitted here. Open3D installs SciPy and scikit-learn automatically. Upstream task registration imports `openai`; evaluation makes no API calls.

Clone the [official VLABench repository](https://github.com/OpenMOSS/VLABench), select the supported revision, install the simulator, and download assets. The asset download script below comes from that official checkout:

```bash
git clone https://github.com/OpenMOSS/VLABench.git third_party/VLABench
git -C third_party/VLABench checkout cf588fe60c0c7282174fe979f5913170cfe69017

pip install mujoco==3.2.2 dm-control==1.0.22 open3d==0.18.0 \
  mediapy==1.2.0 'colorlog>=6,<7' 'openai>=1,<3' \
  gym==0.26.2 ipdb gdown 'websockets>=14,<17' \
  'rrt-algorithms @ git+https://github.com/motion-planning/rrt-algorithms.git@14e47963780affe3ff3bb9fc33a4b14e1a733e85'
pip install --no-deps -e third_party/VLABench
MUJOCO_GL=osmesa python third_party/VLABench/scripts/download_assets.py

```

The default checkout is `third_party/VLABench`; override `EVALUATION.vlabench_root` if you use another location. Both model serving and simulation use Python from the current environment; no `EVALUATION.sim_python` override is needed.

The default OSMesa software renderer requires a system OSMesa library. On Ubuntu/Debian, install it with `sudo apt-get install -y libosmesa6`. Simulator assets require several GB and are separate from training data. Download training data and statistics from `OpenMOSS-Team/vlabench-lerobot-v3.0` following the [data guide](../data/vlabench.md).

## Configuration and execution

Generate the official task inventory first. This checks the checkout and task configurations without a checkpoint, training data, or simulator startup:

```bash
python experiments/vlabench/run_vlabench_manager.py \
  MULTIRUN.create_only=true \
  EVALUATION.output_dir=evaluate_results/vlabench/inventory
```

Set `VLABENCH_CHECKPOINT` to an actual trained `.pt` weights file before evaluation, for example with `export VLABENCH_CHECKPOINT="your actual weights file path"`, then run:

```bash
python experiments/vlabench/run_vlabench_manager.py \
  task=vlabench_easywam_mot_wan22 \
  ckpt="${VLABENCH_CHECKPOINT}" \
  EVALUATION.dataset_stats_path=./data/vlabench-lerobot-v3.0/dataset_stats.json \
  EVALUATION.output_dir=evaluate_results/vlabench/my_run
```

The task must match the trained architecture; choose one of the five existing `vlabench_easywam_*_wan22` recipes. This example uses the downloaded dataset statistics. If training used different statistics, set `EVALUATION.dataset_stats_path` to that run's matching file.

Defaults use one GPU with one model process and two simulator actors, with inference batch 2. Adjust `MULTIRUN.num_gpus`, `MULTIRUN.gpu_ids`, `MULTIRUN.env_num_per_worker`, and `MULTIRUN.inference_batch_size` as needed. For a single-task connectivity check, append `'EVALUATION.tracks=[track_2_cross_category]' 'EVALUATION.tasks=[insert_flower]' EVALUATION.num_trials=1` to the evaluation command. This still requires the real model and simulator and does not replace full evaluation.

## Evaluation scope and results

Defaults select official fixed tracks 1/2/3/4/6, using the first 10 configurations uniformly for each task, totaling 500 episodes. The pinned revision provides only 10 configurations for track 2 / `insert_flower`. Following the official evaluator's precondition, insufficient configurations cause an error rather than truncation, repetition, or skipped tasks. Set `EVALUATION.num_trials=50` only for selections with sufficient configurations, and report evaluation counts when comparing scores. Custom track 5 requires `EVALUATION.tracks=[track_5_cross_task]` and `EVALUATION.custom_track_file`.

Keep `EVALUATION.episode_horizon=null` for official task horizons. Evaluation preserves official absolute EEF actions, IK, one environment step per action, termination, and scoring semantics. Model startup validates checkpoint keys and tensor shapes and rejects incomplete policy weights.

Outputs include `manifest.json`, `tasks.json`, per-episode JSON, logs, `summary.json`, and `summary.csv`. SR/PS/IS denote success, progress, and intention scores. Scores average within tasks, then within tracks, then across tracks. Incomplete coverage produces null formal metrics with separate partial metrics and coverage counts.

Reuse the same `EVALUATION.output_dir` to resume complete results. Use a new directory when weights, statistics, or evaluation configuration change. Simulator or communication failures receive at most two attempts. Set `EVALUATION.video_mode=failures` or `all` to save rollout videos.

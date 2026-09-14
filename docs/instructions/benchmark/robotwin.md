# RoboTwin Evaluation Guide

[中文](robotwin_zh.md) | [Benchmark index](README.md) | [Data and training guide](../data/robotwin.md) | [Back to project README](../../../README.md)

## Installation

Clone the current RoboTwin repository and its XPolicyLab submodule:

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
```

Use Python 3.10 and install the evaluation dependencies not provided by EasyWAM:

```bash
pip install scipy==1.10.1 transforms3d==0.4.2 sapien==3.0.0b1 \
  mplib==0.2.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 \
  "pydantic>=2.5" "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8"

git clone https://github.com/NVlabs/curobo.git third_party/RoboTwin/envs/curobo
pip install --no-build-isolation -e third_party/RoboTwin/envs/curobo
```

RoboTwin rendering uses Vulkan and saves evaluation videos with the system `ffmpeg` executable. On Ubuntu, install:

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg unzip
```

## Assets

Download and extract the official assets, then update their paths:

```bash
huggingface-cli download TianxingChen/RoboTwin2.0 \
  background_texture.zip embodiments.zip objects.zip \
  --repo-type dataset \
  --local-dir third_party/RoboTwin/assets

(
  cd third_party/RoboTwin/assets
  unzip -q -o background_texture.zip
  unzip -q -o embodiments.zip
  unzip -q -o objects.zip
)
(cd third_party/RoboTwin && python scripts/update_embodiment_config_path.py)
```

The clean, randomized, camera, embodiment, and task-limit configurations are included in the cloned repository under `env_cfg/task_config/`.

## Evaluation

Evaluate every task listed by RoboTwin's `_eval_step_limit.yml`:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=./data/robotwin2.0-lerobot-v3.0/dataset_stats.json \
  MULTIRUN.num_gpus=3 'MULTIRUN.gpu_ids=[0,2,4,6]' \
  MULTIRUN.workers_per_gpu=2 MULTIRUN.env_num_per_worker=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10
```

Evaluate one task or override its instruction split:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  EVALUATION.task_name=beat_block_hammer \
  EVALUATION.instruction_type=seen
```

The manager evaluates `demo_clean` and `demo_randomized` sequentially for each task. With no instruction override, it uses the split declared by each current RoboTwin task config. `EVALUATION.eval_num_episodes` controls the number of episodes per phase, while `EVALUATION.replan_steps` controls how many predicted actions are executed before replanning.

With `MULTIRUN.gpu_ids=null`, the first `num_gpus` devices are used; otherwise the first `num_gpus` entries of the ordered `gpu_ids` candidate pool are selected. Model workers are distributed across the selected GPUs in round-robin order. Every worker loads EasyWAM independently; its rollout clients use isolated XPolicyLab sessions and share that worker's dynamic inference batcher. Increasing `workers_per_gpu` increases model memory use. Results remain under `evaluate_results/robotwin/<checkpoint-tag>/<timestamp>/`, including the upstream artifacts, per-phase result files, worker logs, `summary.json`, and `summary.csv`. Reusing the same `EVALUATION.output_dir` resumes phases whose result file is not yet valid.

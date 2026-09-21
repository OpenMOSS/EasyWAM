# RoboDojo Evaluation Guide

[中文](robodojo_zh.md) | [Benchmark index](README.md) | [Data and training](../data/robodojo.md) | [Project README](../../../README.md)

The evaluator runs RoboDojo's 42 simulation tasks with seeds 0, 1, and 2. The 12 Generalization tasks each use 25 standard and 25 `_random` episodes per seed; the other tasks use 50 episodes per seed.

## Setup

Clone the official simulator and its submodules, then prepare its Isaac Sim environment and assets following the [official installation guide](https://github.com/RoboDojo-Benchmark/RoboDojo):

```bash
git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git third_party/RoboDojo
(cd third_party/RoboDojo && bash scripts/install.sh)
(cd third_party/RoboDojo && bash scripts/init_assets.sh)
(cd third_party/RoboDojo && bash scripts/robodojo.sh doctor)
```

Run the manager from the EasyWAM Python environment. The simulator clients use the separate `RoboDojo` conda environment by default. The manager installs its small `easywam_policy` client adapter into the RoboDojo checkout on first use; it will not overwrite a different adapter already present there.

## Evaluation

Run the complete benchmark:

```bash
python experiments/robodojo/run_robodojo_manager.py \
  task=robodojo_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=./data/robodojo-lerobot-v3.0/dataset_stats.json
```

To generate the selected task/seed list without loading a checkpoint or starting Isaac Sim, use `MULTIRUN.create_only=true`. The manager writes all selected jobs to `tasks.jsonl` under the output directory; `MULTIRUN.task_file=<path>` changes that list's destination. During evaluation it separately creates a pending-only worker list after checking existing results.

```bash
python experiments/robodojo/run_robodojo_manager.py \
  EVALUATION.task_name=cover_blocks 'EVALUATION.seeds=[0]' \
  MULTIRUN.create_only=true
```

The default uses eight GPUs for both model and simulator workers, one model copy and one Isaac Sim process per GPU. Each simulator is pinned to its assigned physical GPU and uses local CUDA device 0. Each Isaac Sim process uses the upstream environment count (normally 10; `_random` layouts are capped by RoboDojo). The XPolicyLab adapter enables batched environments, while concurrent clients share each model worker's dynamic inference batcher. Tune `MULTIRUN.inference_batch_size`, `MULTIRUN.inference_batch_wait_ms`, and `EVALUATION.sim_num_envs` for the available memory.

Choose separate GPU pools when desired:

```bash
python experiments/robodojo/run_robodojo_manager.py \
  task=robodojo_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  MULTIRUN.policy_num_gpus=2 'MULTIRUN.policy_gpu_ids=[0,1]' \
  MULTIRUN.env_num_gpus=4 'MULTIRUN.env_gpu_ids=[2,3,4,5]'
```

For a one-episode connectivity test on one task:

```bash
python experiments/robodojo/run_robodojo_manager.py \
  task=robodojo_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.task_name=cover_blocks 'EVALUATION.seeds=[0]' \
  EVALUATION.eval_num_episodes=1 MULTIRUN.num_gpus=1
```

The official `_result.json` and per-camera videos stay in `third_party/RoboDojo/eval_result/RoboDojo/`. EasyWAM writes logs, the resolved configuration, `summary.json`, and `summary.csv` to `evaluate_results/robodojo/<task>/<timestamp>/`. Summary `success_rate` and `score` values are percentages (0–100), matching RoboDojo's official report. A complete official run also refreshes RoboDojo's `_summary.md`. Reuse `EVALUATION.output_dir=<existing-directory>` to resume; jobs with a complete native result are skipped. Use `EVALUATION.sim_env=null` only if the current Python environment already contains RoboDojo and Isaac Sim.

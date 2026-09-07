# RoboTwin Evaluation Guide

[中文](robotwin_zh.md) | [Benchmark index](README.md) | [Data and training guide](../data/robotwin.md) | [Back to project README](../../../README.md)

EasyWAM expects the benchmark at `third_party/RoboTwin`. Follow the official [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) instructions to install the simulator and download its assets. The evaluation worker automatically creates or refreshes the `easywam_policy` symlink inside RoboTwin.

Run all tasks listed by RoboTwin's `_eval_step_limit.yml`:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=./data/robotwin2.0-lerobot-v3.0/dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.env_num_per_gpu=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10
```

Evaluate one task or change the language protocol with overrides:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  EVALUATION.task_name=beat_block_hammer \
  EVALUATION.instruction_type=seen
```

The manager evaluates both `demo_clean` and `demo_randomized` for each task and defaults to unseen instructions. `EVALUATION.eval_num_episodes` controls episodes per phase.

`EVALUATION.skip_get_obs_within_replan=true` skips RGB rendering while the remaining actions in a predicted chunk are executed. This speeds up evaluation, but saved video appears low frame-rate. Set it to `false` for fully rendered video. `EVALUATION.replan_steps` controls the action chunk executed before replanning.

Each GPU runs one persistent model server. Rollout clients claim tasks dynamically, use isolated action-queue sessions, and share server-side inference batches; a task's clean and randomized phases remain sequential. Results are stored under `evaluate_results/robotwin/<checkpoint-tag>/<timestamp>/`, with per-phase result files, worker logs, `summary.json`, and `summary.csv`. A task is skipped on resume only after both phases are valid; reuse the same `EVALUATION.output_dir` timestamp component to resume.

# RoboTwin Guide

[中文](robotwin_zh.md) | [Back to README](../../README.md)

This guide covers RoboTwin 2.0 data preparation, training, and evaluation in EasyWAM.

## Environment and Data

EasyWAM includes its evaluation adapter under `experiments/robotwin/` and expects the benchmark at `third_party/RoboTwin`. Follow the official [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) instructions to finish environment installation and download simulator assets. The evaluation worker automatically creates or refreshes the `easywam_policy` symlink inside RoboTwin; no manual symlink step is required.

Download the split archives from [robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam), then concatenate and extract them:

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
```

The default `configs/data/robotwin.yaml` expects:

```text
data/robotwin2.0/
├── dataset_stats.json
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

The pipeline combines the high camera and two wrist cameras into a 384×320 video. It retains all 33 action/state steps and decodes 9 sparse video timestamps.

## Training

Precompute the RoboTwin text cache. Each prompt is written to its own SHA-256-named file and loaded on demand during training:

```bash
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_wan22
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_cosmos25
```

Select a current task:

| Model | Wan full | Wan LoRA | Cosmos full | Cosmos LoRA |
| --- | --- | --- | --- | --- |
| EasyWAM-MoT | `robotwin_easywam_mot_wan22` | `robotwin_easywam_mot_wan22_lora` | `robotwin_easywam_mot_cosmos25` | `robotwin_easywam_mot_cosmos25_lora` |
| EasyWAM-Unified | `robotwin_easywam_unified_wan22` | `robotwin_easywam_unified_wan22_lora` | `robotwin_easywam_unified_cosmos25` | `robotwin_easywam_unified_cosmos25_lora` |
| EasyWAM-Hidden | `robotwin_easywam_hidden_wan22` | `robotwin_easywam_hidden_wan22_lora` | `robotwin_easywam_hidden_cosmos25` | `robotwin_easywam_hidden_cosmos25_lora` |

For example:

```bash
NPROC_PER_NODE=8 bash scripts/train_zero2.sh task=robotwin_easywam_mot_wan22

NPROC_PER_NODE=8 bash scripts/train_zero2.sh \
  task=robotwin_easywam_mot_cosmos25
```

The default data config loads normalization statistics from `data/robotwin2.0/dataset_stats.json`. Set `data.train.pretrained_norm_stats=null` and `data.val.pretrained_norm_stats=null` if the statistics must be recomputed for a different dataset.

## Evaluation

Run all tasks listed by RoboTwin's `_eval_step_limit.yml`:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
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

Persistent workers keep the model loaded while processing their task shard. Results are stored under `evaluate_results/robotwin/<checkpoint-tag>/<timestamp>/`, with per-phase result files, worker logs, `summary.json`, and `summary.csv`. A task is skipped on resume only after both its clean and randomized phase results are valid; reuse the same `EVALUATION.output_dir` timestamp component to resume.

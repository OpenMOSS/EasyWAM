# LIBERO-Pro Evaluation Guide

[中文](libero_pro_zh.md) | [Benchmark index](README.md) | [LIBERO data guide](../data/libero.md) | [Back to project README](../../../README.md)

LIBERO-Pro evaluates checkpoints trained on standard LIBERO under five perturbations: Object, Position, Semantic, Task, and Environment. The default official protocol covers four base suites, 20 suite/perturbation cells, 200 tasks, and 50 trials per task, for 10,000 episodes total.

## Installation

Clone the official repository into EasyWAM's `third_party` directory:

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git \
  third_party/LIBERO-PRO
```

LIBERO-Pro uses the same simulator dependencies as LIBERO. In the EasyWAM environment, install the packages needed by evaluation:

```bash
pip install mujoco easydict robosuite bddl future \
  cloudpickle gym huggingface_hub
```

Do not run `pip install -e third_party/LIBERO-PRO`. LIBERO-Pro and standard LIBERO use the same Python package name; the dedicated manager loads the requested checkout directly so it does not replace the LIBERO package used by other benchmarks.

## Download BDDL and initial states

The leaderboard BDDL and initial-state files are distributed separately. Following the LIBERO-Pro repository layout, first download them to a temporary directory inside the checkout:

```bash
hf download zhouxueyang/LIBERO-Pro \
  --repo-type dataset \
  --local-dir third_party/LIBERO-PRO/libero_data
```

Then move them into the directories required by LIBERO-Pro:

```bash
mv third_party/LIBERO-PRO/libero_data/bddl_files/* \
  third_party/LIBERO-PRO/libero/libero/bddl_files/
mv third_party/LIBERO-PRO/libero_data/init_files/* \
  third_party/LIBERO-PRO/libero/libero/init_files/
```

The resulting layout must include at least:

```text
third_party/LIBERO-PRO/
└── libero/libero/
    ├── bddl_files/
    │   ├── libero_goal_object/
    │   ├── libero_goal_swap/
    │   └── ...
    └── init_files/
        ├── libero_goal_object/
        ├── libero_goal_swap/
        └── ...
```

EasyWAM's defaults correspond to this layout:

```yaml
EVALUATION:
  libero_pro_root: ./third_party/LIBERO-PRO
  libero_pro_data_root: ${EVALUATION.libero_pro_root}/libero/libero
```

The manager loads `libero_pro_root`, reads BDDL and initial states from `libero_pro_data_root`, and creates an isolated LIBERO path configuration in the evaluation output directory. It does not rewrite the upstream checkout. If the repository is cloned elsewhere, override `EVALUATION.libero_pro_root`; the data path follows it automatically unless explicitly overridden.

## Validation and evaluation

Validate the source, assets, all 20 suites, and every task file without loading a model or using a GPU:

```bash
python experiments/libero_pro/run_libero_pro_manager.py \
  MULTIRUN.create_only=true
```

Run the complete official evaluation:

```bash
python experiments/libero_pro/run_libero_pro_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  MULTIRUN.num_gpus=4
```

For example, when the repository is stored elsewhere:

```bash
python experiments/libero_pro/run_libero_pro_manager.py \
  MULTIRUN.create_only=true \
  EVALUATION.libero_pro_root=/path/to/LIBERO-PRO
```

Filter base suites, perturbations, and zero-based task IDs during development:

```bash
python experiments/libero_pro/run_libero_pro_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial]' \
  'MULTIRUN.perturbations=[object,semantic]' \
  'MULTIRUN.task_ids=[0]'
```

Valid perturbations are `object`, `position`, `semantic`, `task`, and `environment`. Task IDs apply to every selected suite/perturbation pair. Official step budgets are Spatial 220, Object 280, Goal 300, and LIBERO-10 520. Every task must provide at least 50 initial states.

## Concurrency, results, and resume behavior

`MULTIRUN.num_gpus`, `gpu_ids`, `workers_per_gpu`, `env_num_per_worker`, `inference_batch_size`, and `inference_batch_wait_ms` have the same meaning as in other EasyWAM managers. Every model worker loads its own model, while its rollout actors share a dynamic inference batcher.

Results are written under `evaluate_results/libero_pro/<task>/<timestamp>/`. Outputs include the task manifest, upstream source metadata, isolated runtime configuration, worker logs, per-task results, error records, `summary.json`, `summary.csv`, `task_results.csv`, and `leaderboard.csv`. The leaderboard uses normalized 0–1 success rates and contains the official four-suite by five-perturbation matrix plus macro averages.

Reusing `EVALUATION.output_dir` skips only results whose protocol metadata and all 50 episodes are complete. Corrupt and incomplete results are evaluated again. Partial runs report coverage, and a leaderboard is marked complete only when every selected cell is complete.

This integration covers the official five-dimension, single-perturbation leaderboard only. It does not generate combined perturbations or include the later seven-case extension.

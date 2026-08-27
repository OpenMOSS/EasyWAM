# LIBERO-Plus Guide

[中文](libero_plus_zh.md) | [Back to README](../README.md)

LIBERO-Plus is a robustness evaluation benchmark. EasyWAM evaluates a checkpoint trained on LIBERO; there is no separate LIBERO-Plus training task.

## Dedicated Environment

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) and vanilla LIBERO install under the same `libero` Python package name. Use a dedicated environment for LIBERO-Plus evaluation, or replace the vanilla package in an existing environment.

After installing EasyWAM, install the fork and its additional system/Python dependencies according to the upstream instructions. A typical setup is:

```bash
sudo apt install libexpat1 libfontconfig1-dev libmagickwand-dev
pip install wand scikit-image
git clone https://github.com/sylvestf/LIBERO-plus.git <path/to/LIBERO-plus>
pip install --no-deps -e <path/to/LIBERO-plus>
```

Download `assets.zip` from the [official LIBERO-Plus assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) and extract it under the fork's `libero/libero/assets/` directory.

`${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml` must point to the LIBERO-Plus `benchmark_root`, `bddl_files`, `init_states`, and `assets` directories. The manager verifies these paths, the extended asset directories, `task_classification.json`, BDDL sources, and official suite sizes before launching workers.

## Checkpoint and Statistics

Use any EasyWAM checkpoint trained with a LIBERO task, for example:

- `libero_easywam_mot` or `libero_easywam_mot_lora`
- `libero_easywam_unified` or `libero_easywam_unified_lora`
- `libero_easywam_hidden` or `libero_easywam_hidden_lora`

`dataset_stats.json` is required. If `EVALUATION.dataset_stats_path` is omitted, the manager searches up to four parent directories of the checkpoint.

Validate the installation and build a task manifest without running rollouts:

```bash
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot \
  MULTIRUN.create_only=true \
  EVALUATION.output_dir=./evaluate_results/libero_plus/validation
```

## Full Evaluation

The official configuration evaluates every selected perturbation task once. The four suites contain 10,030 tasks in total.

```bash
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
```

Filter a development run by suite, perturbation category, difficulty, or zero-based task ID:

```bash
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial]' \
  'MULTIRUN.categories=[camera,light]' \
  'MULTIRUN.difficulty_levels=[1,2]' \
  'MULTIRUN.task_ids=[0,1,2]'
```

Valid category slugs are `layout`, `camera`, `robot`, `language`, `light`, `background`, and `noise`; difficulty levels range from 1 to 5. Task IDs apply to every selected suite.

The manager creates persistent model workers and automatically switches between EGL and OSMesa based on the number of workers per GPU. Videos and progress output are disabled by default.

Results are written to `evaluate_results/libero_plus/<task>/<timestamp>/`. The directory includes `tasks.jsonl`, worker logs, per-task results, errors, `summary.json`, `summary.csv`, and `task_results.csv`, with aggregate statistics by suite, perturbation category, difficulty, and category-by-difficulty. Reuse the same explicit `EVALUATION.output_dir` to skip valid results and resume an interrupted run.

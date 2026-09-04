# EasyWAM Configuration Guide

[中文](README_zh.md) | [Back to README](../../README.md)

EasyWAM uses [Hydra](https://hydra.cc/) to compose training, data, model, and task configuration. This directory explains how the configuration groups fit together and how to tune them without copying an entire task file.

## Configuration layout

```text
configs/
├── train.yaml                 # Global training and model-execution defaults
├── data/                      # Dataset, processor, cameras, and normalization
├── model/
│   ├── backbone/              # Backbone paths, dimensions, and attention settings
│   ├── lora/                  # LoRA adapter settings
│   └── easywam_*.yaml         # EasyWAM architecture definitions
├── task/                      # Ready-to-run data + model + training recipes
└── sim_*.yaml                 # Benchmark evaluation defaults
```

`configs/train.yaml` is the root training configuration. A task config overrides its `data` and `model` groups and then applies benchmark-specific training values. Model configs compose a backbone and, for LoRA recipes, a LoRA config.

```text
train.yaml
└── task=libero_easywam_mot_wan22_lora
    ├── data=libero_2cam
    └── model=easywam_mot_wan22_lora
        ├── backbone=wan22
        └── lora=video_dit
```

Use a task recipe for normal training and evaluation. Override `data=` or `model=` directly only when building a new recipe and after checking that action/state dimensions and backbone dimensions remain compatible.

## Inspect and override configuration

Print the fully composed training configuration without starting a run:

```bash
python scripts/train.py --cfg job task=libero_easywam_mot_wan22
```

Hydra overrides use dotted keys. Later command-line values take precedence over values composed from `train.yaml` and the task:

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_wan22 \
  batch_size=12 \
  learning_rate=5e-5 \
  model.loss.lambda_action=2.0 \
  model.backbone.attention_backend=auto
```

Quote list overrides so that the shell does not interpret brackets:

```bash
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot_wan22 \
  'MULTIRUN.task_suite_names=[libero_spatial]' \
  'MULTIRUN.categories=[camera,light]'
```

Hydra does not change the working directory in this project (`hydra.job.chdir=false`), so relative dataset, checkpoint, cache, and output paths are resolved from the directory where the command is launched. Run commands from the repository root unless every relative path is adjusted.

## Topics

| Topic | English | 中文 |
| --- | --- | --- |
| Models, backbones, LoRA, and losses | [Model configuration](models.md) | [模型配置](models_zh.md) |
| Data, training, logging, and evaluation | [Training configuration](training.md) | [训练配置](training_zh.md) |
| Throughput, memory, and inference latency | [Efficiency configuration](efficiency.md) | [效率配置](efficiency_zh.md) |

Benchmark installation, dataset layout, task selection, and result-resume behavior are covered by the [LIBERO](../instructions/libero.md), [LIBERO-Plus](../instructions/libero_plus.md), and [RoboTwin](../instructions/robotwin.md) guides.

## Recommended workflow

1. Choose the closest task recipe from `configs/task/`.
2. Update checkpoint and dataset paths through command-line overrides or a new task recipe.
3. Inspect the composed configuration with `--cfg job`.
4. Precompute text embeddings with the same task selection used for training.
5. Start from the checked-in defaults, then tune memory and throughput settings for the target hardware.

Keep reusable experiment choices in a task YAML. Reserve command-line overrides for paths, short experiments, and values intentionally varied between runs.

# LIBERO 使用指南

[English](libero.md) | [返回 README](../README_zh.md)

本文介绍 EasyWAM 中 LIBERO 的数据准备、训练与评测流程。

## 环境与数据

先安装 EasyWAM，再在同一环境中安装官方 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 包。当前配置使用的预处理数据基于 MuJoCo 3.3.2，建议保持仿真器版本一致：

```bash
pip install mujoco==3.3.2
```

从 [LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) 下载四个压缩包并解压：

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2
for archive in *.tar.gz; do
  tar -xzf "$archive"
done
cd ../..
```

默认的 `configs/data/libero_2cam.yaml` 使用以下目录：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

数据管线会在 224 px 分辨率下横向拼接 agent 和 wrist 两个相机，保留全部 33 个 action/state 时间步，同时仅解码 `[0, 4, ..., 32]` 对应的 9 帧视频。

## 训练

首先预计算一次聚合式文本 embedding cache。所有 LIBERO 模型配置都可以共用该 cache：

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot
```

也可以使用多张 GPU：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py task=libero_easywam_mot
```

当前有效的任务名如下：

| 模型 | 全参数训练 | LoRA |
| --- | --- | --- |
| EasyWAM-MoT | `libero_easywam_mot` | `libero_easywam_mot_lora` |
| EasyWAM-Unified | `libero_easywam_unified` | `libero_easywam_unified_lora` |
| EasyWAM-Hidden | `libero_easywam_hidden` | `libero_easywam_hidden_lora` |

将 task 作为 Hydra override 启动训练：

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot
```

需要 ZeRO-2 或 ZeRO-2 CPU Offload 时，分别使用 `scripts/train_zero2.sh` 或 `scripts/train_zero2_offload.sh`。训练结果保存在 `runs/<task>/<run-id>/`。如果没有配置预先计算的归一化统计，首次训练会在 run 目录生成 `dataset_stats.json`；评测时应使用与 checkpoint 匹配的统计文件。

## 评测

启动 manager 前需要安装官方 LIBERO 仿真环境。评测自己训练的 checkpoint：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot \
  ckpt=./runs/libero_easywam_mot/<run-id>/checkpoints/weights/<checkpoint>.pt \
  EVALUATION.dataset_stats_path=./runs/libero_easywam_mot/<run-id>/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

常用参数示例：

```bash
# 只评测部分 suite，在四张 GPU 上各启动一个 worker
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object]' \
  MULTIRUN.num_gpus=4 MULTIRUN.max_tasks_per_gpu=1

# 仅检查安装并生成任务清单，不启动 rollout
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot ckpt=<path/to/checkpoint.pt> MULTIRUN.create_only=true
```

默认协议会评测四个 suite，每个任务执行 50 次。每张 GPU 一个 worker 时 manager 使用 EGL；单卡多 worker 时使用 OSMesa。默认关闭视频和进度渲染，可通过 `EVALUATION.video_mode`、`EVALUATION.visualize_future_video` 和 `EVALUATION.progress` 调整。

结果保存在 `evaluate_results/libero/<task>/<timestamp>/`，其中包括 worker 日志、逐任务 JSON、`summary.json`、`summary.csv` 和 `task_success_rates.csv`。重新指定同一个 `EVALUATION.output_dir` 即可续评，manager 会跳过结果完整的任务。

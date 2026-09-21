# LIBERO 数据与训练指南

[English](libero.md) | [数据索引](README_zh.md) | [评测指南](../benchmark/libero_zh.md) | [返回项目 README](../../../README_zh.md)

本文介绍 EasyWAM 中 LIBERO 的训练数据准备与训练流程。

## 训练数据

从 [OpenMOSS-Team/libero-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/libero-lerobot-v3.0) 下载可直接使用的 LeRobot v3.0 数据集。下载位置与 `configs/data/libero.yaml` 中的默认路径一致：

```bash
huggingface-cli download OpenMOSS-Team/libero-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/libero-lerobot-v3.0
```

默认的 `configs/data/libero.yaml` 使用以下目录：

```text
data/libero-lerobot-v3.0/
├── libero_10/
├── libero_goal/
├── libero_object/
└── libero_spatial/
```

EasyWAM 仍兼容 LeRobot v2.1 数据，并为需要转换到 v3.0 的用户提供了 `scripts/convert_lerobot_v21_to_v30.py`。

数据管线会在 224 px 分辨率下横向拼接 agent 和 wrist 两个相机，保留全部 33 个 action/state 时间步，同时仅解码 `[0, 4, ..., 32]` 对应的 9 帧视频。

## 训练

首先预计算一次逐指令文本 embedding cache。每条 prompt 按 SHA-256 哈希单独存储，使用相同文本 encoder 的模型配置可以共用该 cache：

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
python scripts/precompute_text_embeds.py task=libero_easywam_mot_cosmos25
```

也可以使用多张 GPU：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
```

当前有效的任务名如下：

| 模型 | Wan 全量 | Wan LoRA | Cosmos 全量 | Cosmos LoRA |
| --- | --- | --- | --- | --- |
| EasyWAM-MoT | `libero_easywam_mot_wan22` | `libero_easywam_mot_wan22_lora` | `libero_easywam_mot_cosmos25` | `libero_easywam_mot_cosmos25_lora` |
| EasyWAM-Unified | `libero_easywam_unified_wan22` | `libero_easywam_unified_wan22_lora` | `libero_easywam_unified_cosmos25` | `libero_easywam_unified_cosmos25_lora` |
| EasyWAM-Hidden | `libero_easywam_hidden_wan22` | `libero_easywam_hidden_wan22_lora` | `libero_easywam_hidden_cosmos25` | `libero_easywam_hidden_cosmos25_lora` |

将 task 作为 Hydra override 启动训练：

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot_wan22

# Cosmos-Predict2.5 backbone（MoT/Hidden/Unified 通用）
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_cosmos25
```

需要 ZeRO-2 或 ZeRO-2 CPU Offload 时，分别使用 `scripts/train_zero2.sh` 或 `scripts/train_zero2_offload.sh`。训练结果保存在 `runs/<task>/<run-id>/`。如果没有配置预先计算的归一化统计，首次训练会在 run 目录生成 `dataset_stats.json`；评测时应使用与 checkpoint 匹配的统计文件。

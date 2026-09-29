# RoboTwin 数据与训练指南

[English](robotwin.md) | [数据索引](README_zh.md) | [评测指南](../benchmark/robotwin_zh.md) | [返回项目 README](../../../README_zh.md)

本文介绍 EasyWAM 中 RoboTwin 2.0 的训练数据准备与训练流程。

## 训练数据

从 [OpenMOSS-Team/robotwin2.0-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/robotwin2.0-lerobot-v3.0) 获取 LeRobot v3.0 数据，并将 clean 与 randomized 数据分别放在 `configs/data/robotwin.yaml` 默认使用的子目录中：

```bash
huggingface-cli download OpenMOSS-Team/robotwin2.0-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/robotwin2.0-lerobot-v3.0
```

默认的 `configs/data/robotwin.yaml` 使用以下目录：

```text
data/robotwin2.0-lerobot-v3.0/
├── clean/                 # 50 个任务 × 50 条轨迹
├── random/                # 50 个任务 × 500 条轨迹
└── dataset_stats.json     # full 模式归一化统计
```

EasyWAM 训练仅支持 LeRobot v3.0 数据集。

数据管线会把 high camera 和两个 wrist camera 组合为 384×320 视频，保留全部 33 个 action/state 时间步，并稀疏解码 9 帧视频。

## 训练

预计算 RoboTwin 文本 cache。每条 prompt 会按 SHA-256 哈希单独写入文件，并在训练时按需加载：

```bash
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_wan22
```

当前提供五种 Wan2.2 task 配置：

| 架构 | Task |
| --- | --- |
| MoT | `robotwin_easywam_mot_wan22` |
| Hidden | `robotwin_easywam_hidden_wan22` |
| Unified | `robotwin_easywam_unified_wan22` |
| MoT-Joint | `robotwin_easywam_mot_joint_wan22` |
| MoT-IDM | `robotwin_easywam_mot_idm_wan22` |

例如：

```bash
# 全量训练（默认）：同时加载 clean 和 random，保持原始 1:10 比例
NPROC_PER_NODE=8 bash scripts/train_zero2.sh \
  task=robotwin_easywam_mot_wan22 data.mode=full

# 仅 clean / 仅 random
NPROC_PER_NODE=8 bash scripts/train_zero2.sh \
  task=robotwin_easywam_mot_wan22 data.mode=clean
NPROC_PER_NODE=8 bash scripts/train_zero2.sh \
  task=robotwin_easywam_mot_wan22 data.mode=random
```

`data.mode` 同时控制训练、验证和文本 embedding 预计算的数据目录。`clean`、`random` 分别使用子目录内的 `dataset_stats.json`；`full` 使用父目录中的全量统计。若使用了不同的数据集并需要重新计算统计，可将 `data.train.pretrained_norm_stats=null` 和 `data.val.pretrained_norm_stats=null` 作为 override 传入。

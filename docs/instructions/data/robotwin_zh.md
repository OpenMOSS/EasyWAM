# RoboTwin 数据与训练指南

[English](robotwin.md) | [数据索引](README_zh.md) | [评测指南](../benchmark/robotwin_zh.md) | [返回项目 README](../../../README_zh.md)

本文介绍 EasyWAM 中 RoboTwin 2.0 的训练数据准备与训练流程。

## 训练数据

从 [OpenMOSS-Team/robotwin2.0-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/robotwin2.0-lerobot-v3.0) 下载可直接使用的 LeRobot v3.0 数据集。下载位置与 `configs/data/robotwin.yaml` 中的默认路径一致：

```bash
huggingface-cli download OpenMOSS-Team/robotwin2.0-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/robotwin2.0-lerobot-v3.0
```

默认的 `configs/data/robotwin.yaml` 使用以下目录：

```text
data/robotwin2.0-lerobot-v3.0/
├── data/
├── dataset_stats.json
├── meta/
└── videos/
```

EasyWAM 仍兼容 LeRobot v2.1 数据，并为需要转换到 v3.0 的用户提供了 `scripts/convert_lerobot_v21_to_v30.py`。

数据管线会把 high camera 和两个 wrist camera 组合为 384×320 视频，保留全部 33 个 action/state 时间步，并稀疏解码 9 帧视频。

## 训练

预计算 RoboTwin 文本 cache。每条 prompt 会按 SHA-256 哈希单独写入文件，并在训练时按需加载：

```bash
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_wan22
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_cosmos25
```

当前有效的任务名如下：

| 模型 | Wan 全量 | Wan LoRA | Cosmos 全量 | Cosmos LoRA |
| --- | --- | --- | --- | --- |
| EasyWAM-MoT | `robotwin_easywam_mot_wan22` | `robotwin_easywam_mot_wan22_lora` | `robotwin_easywam_mot_cosmos25` | `robotwin_easywam_mot_cosmos25_lora` |
| EasyWAM-Unified | `robotwin_easywam_unified_wan22` | `robotwin_easywam_unified_wan22_lora` | `robotwin_easywam_unified_cosmos25` | `robotwin_easywam_unified_cosmos25_lora` |
| EasyWAM-Hidden | `robotwin_easywam_hidden_wan22` | `robotwin_easywam_hidden_wan22_lora` | `robotwin_easywam_hidden_cosmos25` | `robotwin_easywam_hidden_cosmos25_lora` |

例如：

```bash
NPROC_PER_NODE=8 bash scripts/train_zero2.sh task=robotwin_easywam_mot_wan22

NPROC_PER_NODE=8 bash scripts/train_zero2.sh \
  task=robotwin_easywam_mot_cosmos25
```

默认数据配置从 `data/robotwin2.0-lerobot-v3.0/dataset_stats.json` 加载归一化统计。如果使用了不同的数据集并需要重新计算统计，可将 `data.train.pretrained_norm_stats=null` 和 `data.val.pretrained_norm_stats=null` 作为 override 传入。

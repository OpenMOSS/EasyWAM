# RoboDojo 数据与训练

[English](robodojo.md) | [数据索引](README_zh.md) | [返回项目 README](../../../README_zh.md)

本文介绍 EasyWAM 中 RoboDojo 的训练数据准备与训练流程。

## 训练数据

从 [OpenMOSS-Team/robodojo-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/robodojo-lerobot-v3.0) 下载可直接使用的 LeRobot v3.0 数据集。下载位置与 `configs/data/robodojo.yaml` 中的默认路径一致：

```bash
huggingface-cli download OpenMOSS-Team/robodojo-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/robodojo_lerobot_v3.0
```

默认的 `configs/data/robodojo.yaml` 使用以下目录：

```text
data/robodojo_lerobot_v3.0/
├── data/
├── dataset_stats.json
├── meta/
└── videos/
```

数据配置读取 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 及 14 维关节状态/动作。每 32 步动作使用 9 帧视频，三相机拼成 384×320 图像；归一化使用全局 z-score，并以固定随机种子留出 1% episode 验证。

## 训练

每个 backbone 预计算一次 RoboDojo 文本 cache。每条 prompt 按 SHA-256 哈希单独存储，使用该 backbone 的任务配置共享缓存：

```bash
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_cosmos25
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_flux2_klein_4b
```

将 task 作为 Hydra override 启动训练，例如：

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=robodojo_easywam_mot_wan22

NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=robodojo_easywam_mot_cosmos25

NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=robodojo_easywam_unified_wan22_lora
```

需要训练其他模型时，换用 `configs/task/` 下对应的 `robodojo_*.yaml` 配方。默认数据配置从 `data/robodojo_lerobot_v3.0/dataset_stats.json` 加载归一化统计。训练结果及其配套统计保存在 `runs/<task>/<run-id>/`。

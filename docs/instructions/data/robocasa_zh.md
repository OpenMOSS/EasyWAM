# RoboCasa365 数据与训练指南

[English](robocasa.md) | [数据索引](README_zh.md) | [评测指南](../benchmark/robocasa_zh.md) | [返回项目 README](../../../README_zh.md)

本文介绍 EasyWAM 中 RoboCasa365 的训练数据准备与训练流程。

## 下载

EMBER 镜像已经是扁平的 LeRobot v3.0 数据集，无需再次转换：

```bash
huggingface-cli download ember-lab-berkeley/robocasa365-pretrain-atomic \
  --repo-type dataset \
  --local-dir data/robocasa365-lerobot-v3.0/pretrain-atomic

huggingface-cli download ember-lab-berkeley/robocasa365-pretrain-composite \
  --repo-type dataset \
  --local-dir data/robocasa365-lerobot-v3.0/pretrain-composite
```

**configs/data/robocasa.yaml** 默认读取：

```text
data/robocasa365-lerobot-v3.0/
├── pretrain-atomic/
└── pretrain-composite/
```

**pretrain** 是覆盖面更广的基础预训练数据；**target** 是 benchmark 任务的 post-training / 微调数据，本次初始训练配置不包含它。

适配器严格保留数据 schema：三路 256 px 图像按 left、right、wrist 横向拼接；state 顺序为 base_pos(3) + base_quat(4) + eef_pos_rel(3) + eef_quat_rel(4) + gripper_qpos(2)；action 顺序为 base_motion(4) + control_mode(1) + eef_delta_pos(3) + eef_delta_axis_angle(3) + gripper(1)。

## 统计与训练

先生成文本特征，再开始训练：

```bash
python scripts/precompute_text_embeds.py task=robocasa_easywam_mot_wan22
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=robocasa_easywam_mot_wan22
```

使用其他模型或 LoRA 时替换为对应 task 配置。数据配置没有指定预计算归一化统计；首次运行时，EasyWAM 会针对 atomic 与 composite 的组合数据计算统计，并保存到：

```text
runs/robocasa_easywam_mot_wan22/<run-id>/dataset_stats.json
```

请让此文件与 checkpoint 配套，并在评测时传入。上游两个数据目录中的空 **meta/stats.json** 不能作为模型归一化统计使用。

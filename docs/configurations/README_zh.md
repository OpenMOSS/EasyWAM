# EasyWAM 配置指南

[English](README.md) | [返回 README](../../README_zh.md)

EasyWAM 使用 [Hydra](https://hydra.cc/) 组合训练、数据、模型和任务配置。本目录说明各配置组的关系，以及如何在不复制整份 task 配置的情况下进行调整。

## 配置目录

```text
configs/
├── train.yaml                 # 全局训练与模型执行默认值
├── data/                      # 数据集、processor、相机和归一化
├── model/
│   ├── backbone/              # Backbone 路径、维度与 Attention 设置
│   ├── lora/                  # LoRA adapter 设置
│   └── easywam_*.yaml         # EasyWAM 架构定义
├── task/                      # 可直接运行的数据、模型和训练组合
└── sim_*.yaml                 # Benchmark 评测默认值
```

`configs/train.yaml` 是训练配置根节点。task 配置会覆盖其中的 `data` 和 `model` 组，再设置特定 Benchmark 的训练参数。模型配置继续组合 backbone；LoRA 训练还会组合 LoRA 配置。

```text
train.yaml
└── task=libero_easywam_mot_wan22_lora
    ├── data=libero_2cam
    └── model=easywam_mot_wan22_lora
        ├── backbone=wan22
        └── lora=video_dit
```

常规训练和评测应优先选择 task 配置。只有在创建新 recipe 时才建议直接覆盖 `data=` 或 `model=`，并需要确认 action/state 维度和 backbone 维度兼容。

## 查看和覆盖配置

不启动训练，打印 Hydra 完整组合结果：

```bash
python scripts/train.py --cfg job task=libero_easywam_mot_wan22
```

Hydra 使用点分键覆盖配置。命令行中靠后的值优先于 `train.yaml` 和 task 组合得到的值：

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_wan22 \
  batch_size=12 \
  learning_rate=5e-5 \
  model.loss.lambda_action=2.0 \
  model.backbone.attention_backend=auto
```

列表 override 应使用引号，避免方括号被 shell 解释：

```bash
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot_wan22 \
  'MULTIRUN.task_suite_names=[libero_spatial]' \
  'MULTIRUN.categories=[camera,light]'
```

本项目设置了 `hydra.job.chdir=false`，Hydra 不会切换工作目录。因此，相对的数据集、checkpoint、cache 和输出路径都从命令启动目录解析。除非已经调整所有相对路径，否则应从仓库根目录运行命令。

## 专题文档

| 专题 | English | 中文 |
| --- | --- | --- |
| 模型、backbone、LoRA 和损失 | [Model configuration](models.md) | [模型配置](models_zh.md) |
| 数据、训练、日志和评测 | [Training configuration](training.md) | [训练配置](training_zh.md) |
| 吞吐、显存和推理延迟 | [Efficiency configuration](efficiency.md) | [效率配置](efficiency_zh.md) |

Benchmark 安装、数据目录、任务选择和断点续评请参阅 [LIBERO](../instructions/libero_zh.md)、[LIBERO-Plus](../instructions/libero_plus_zh.md) 和 [RoboTwin](../instructions/robotwin_zh.md) 指南。

## 推荐流程

1. 从 `configs/task/` 选择最接近需求的 task recipe。
2. 通过命令行 override 或新 task recipe 调整 checkpoint 和数据路径。
3. 使用 `--cfg job` 检查完整组合配置。
4. 使用与训练相同的 task 选择预计算文本 embedding。
5. 先使用仓库默认值，再针对目标硬件调整显存和吞吐配置。

可复用的实验选择应保存在 task YAML 中。命令行 override 更适合路径、短期实验以及需要在多次运行间主动变化的参数。

# RoboTwin 使用指南

[English](robotwin.md) | [返回 README](../README_zh.md)

本文介绍 EasyWAM 中 RoboTwin 2.0 的数据准备、训练与评测流程。

## 环境与数据

EasyWAM 的评测适配代码位于 `experiments/robotwin/`，默认 benchmark 目录为 `third_party/RoboTwin`。请按照官方 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 指南完成环境安装并下载仿真资源。评测 worker 会自动在 RoboTwin 中创建或刷新 `easywam_policy` 软链接，不需要手动执行链接命令。

从 [robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam) 下载所有分卷并合并解压：

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
```

默认的 `configs/data/robotwin.yaml` 使用以下目录：

```text
data/robotwin2.0/
├── dataset_stats.json
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

数据管线会把 high camera 和两个 wrist camera 组合为 384×320 视频，保留全部 33 个 action/state 时间步，并稀疏解码 9 帧视频。

## 训练

预计算所有 RoboTwin 模型共享的文本 cache：

```bash
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot
```

当前有效的任务名如下：

| 模型 | 全参数训练 | LoRA |
| --- | --- | --- |
| EasyWAM-MoT | `robotwin_easywam_mot` | `robotwin_easywam_mot_lora` |
| EasyWAM-Unified | `robotwin_easywam_unified` | `robotwin_easywam_unified_lora` |
| EasyWAM-Hidden | `robotwin_easywam_hidden` | `robotwin_easywam_hidden_lora` |

例如：

```bash
NPROC_PER_NODE=8 bash scripts/train_zero2.sh task=robotwin_easywam_mot
```

默认数据配置从 `data/robotwin2.0/dataset_stats.json` 加载归一化统计。如果使用了不同的数据集并需要重新计算统计，可将 `data.train.pretrained_norm_stats=null` 和 `data.val.pretrained_norm_stats=null` 作为 override 传入。

## 评测

评测 RoboTwin `_eval_step_limit.yml` 中列出的全部任务：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
```

可以通过 override 只评测一个任务或切换语言指令协议：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  EVALUATION.task_name=beat_block_hammer \
  EVALUATION.instruction_type=seen
```

manager 会对每个任务分别评测 `demo_clean` 和 `demo_randomized`，默认使用 unseen instruction。每个阶段的 episode 数量由 `EVALUATION.eval_num_episodes` 控制。

`EVALUATION.skip_get_obs_within_replan=true` 会在连续执行一次预测 action chunk 的剩余动作时跳过 RGB 渲染，从而加速评测，但保存的视频会显得帧率很低。如果需要完整渲染视频，请设置为 `false`。`EVALUATION.replan_steps` 控制每次重新规划前执行的动作数。

常驻 worker 会在处理任务分片时保持模型已加载状态。结果保存在 `evaluate_results/robotwin/<checkpoint-tag>/<timestamp>/`，其中包括各阶段结果文件、worker 日志、`summary.json` 和 `summary.csv`。只有 clean 和 randomized 两个阶段的结果都有效时，续评才会跳过该任务；使用相同的 `EVALUATION.output_dir` 时间戳部分即可继续。

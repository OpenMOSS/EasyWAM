# VLABench 数据与训练

[English](vlabench.md) | [数据索引](README_zh.md) | [评测指南](../benchmark/vlabench_zh.md)

以下命令在项目根目录执行。

## 训练数据

从 [OpenMOSS-Team/vlabench-lerobot-v3.0](https://huggingface.co/datasets/OpenMOSS-Team/vlabench-lerobot-v3.0) 下载可直接使用的 LeRobot v3.0 数据集，无需再次转换：

```bash
hf download OpenMOSS-Team/vlabench-lerobot-v3.0 \
  --repo-type dataset \
  --local-dir data/vlabench-lerobot-v3.0
```

下载位置与 `configs/data/vlabench.yaml` 的默认路径一致：

```text
data/vlabench-lerobot-v3.0/
├── data/
├── dataset_stats.json
├── meta/
├── README.md
└── videos/
```

数据包含 5000 个 episode、575101 帧，采样率为 10 FPS。三路相机为 `image`、`wrist_image`、`second_image`，拼接为 384×320 图像。状态和动作为 10 维 `[xyz, rotation6d, gripper]`，位置与姿态采用机器人基座坐标系，rotation6d 为旋转矩阵前两行；state 的夹爪值 1 表示闭合，action 的 1 表示张开。

默认每 32 步动作使用 9 帧视频，从数据集根目录的 `dataset_stats.json` 加载全局 min/max 归一化统计，未留出独立验证集。训练复用现有 LeRobot 数据加载器和 WAMProcessor，无需新增训练依赖。

## 训练

先按项目 README 准备 Wan2.2 模型权重，然后预计算文本缓存：

```bash
python scripts/precompute_text_embeds.py task=vlabench_easywam_mot_wan22
```

五种 Wan2.2 配方共享文本缓存，分别为 `mot`、`hidden`、`unified`、`mot_joint`、`mot_idm`，命名为 `vlabench_easywam_<architecture>_wan22`，均可在 `configs/task/` 下找到。缓存精度须与训练一致。

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=vlabench_easywam_mot_wan22
```

根据可用 GPU 数调整 `NPROC_PER_NODE`。默认每进程 batch=4、BF16、学习率 1e-4，训练 30000 步，每 1000 步保存。训练结果及配套统计保存在 `runs/<task>/<run-id>/`。评测时使用与该训练任务匹配的 checkpoint 和归一化统计，见[评测指南](../benchmark/vlabench_zh.md)。

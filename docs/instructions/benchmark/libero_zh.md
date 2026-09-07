# LIBERO 评测指南

[English](libero.md) | [Benchmark 索引](README_zh.md) | [数据与训练指南](../data/libero_zh.md) | [返回项目 README](../../../README_zh.md)

请在同一环境中安装 EasyWAM 和官方 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 仿真器。项目使用 MuJoCo 3.3.2：

```bash
pip install mujoco==3.3.2
```

使用训练 checkpoint 及其对应的归一化统计进行评测：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=./runs/libero_easywam_mot_wan22/<run-id>/checkpoints/weights/<checkpoint>.pt \
  EVALUATION.dataset_stats_path=./runs/libero_easywam_mot_wan22/<run-id>/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

常用参数示例：

```bash
# 只评测部分 suite，每张 GPU 运行四个环境
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object]' \
  MULTIRUN.num_gpus=4 MULTIRUN.env_num_per_gpu=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10

# 仅检查安装并生成任务清单，不启动 rollout
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> MULTIRUN.create_only=true
```

默认协议会评测四个 suite，每个任务执行 50 次。每张 GPU 只加载一个模型，环境动态领取任务并共享推理组批器。单环境使用 EGL，并发环境使用 OSMesa。默认关闭视频和进度渲染，可通过 `EVALUATION.video_mode`、`EVALUATION.visualize_future_video` 和 `EVALUATION.progress` 调整。

结果保存在 `evaluate_results/libero/<task>/<timestamp>/`，其中包括 worker 日志、逐任务 JSON、`summary.json`、`summary.csv` 和 `task_success_rates.csv`。重新指定同一个 `EVALUATION.output_dir` 即可续评，manager 会跳过结果完整的任务。

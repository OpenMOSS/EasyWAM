# RoboTwin 评测指南

[English](robotwin.md) | [Benchmark 索引](README_zh.md) | [数据与训练指南](../data/robotwin_zh.md) | [返回项目 README](../../../README_zh.md)

EasyWAM 默认在 `third_party/RoboTwin` 查找 benchmark。请按照官方 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 指南安装仿真环境并下载资源。评测 worker 会自动在 RoboTwin 中创建或刷新 `easywam_policy` 软链接。

评测 RoboTwin `_eval_step_limit.yml` 中列出的全部任务：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=./data/robotwin2.0-lerobot-v3.0/dataset_stats.json \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.env_num_per_gpu=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10
```

可以通过 override 只评测一个任务或切换语言指令协议：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  EVALUATION.task_name=beat_block_hammer \
  EVALUATION.instruction_type=seen
```

manager 会对每个任务分别评测 `demo_clean` 和 `demo_randomized`，默认使用 unseen instruction。每个阶段的 episode 数量由 `EVALUATION.eval_num_episodes` 控制。

`EVALUATION.skip_get_obs_within_replan=true` 会在连续执行一次预测 action chunk 的剩余动作时跳过 RGB 渲染，从而加速评测，但保存的视频会显得帧率很低。如果需要完整渲染视频，请设置为 `false`。`EVALUATION.replan_steps` 控制每次重新规划前执行的动作数。

每张 GPU 运行一个常驻模型服务。rollout 客户端动态领取任务，使用隔离的动作队列会话，并共享服务端推理 batch；同一任务的 clean 和 randomized 阶段仍按顺序执行。结果保存在 `evaluate_results/robotwin/<checkpoint-tag>/<timestamp>/`，其中包括各阶段结果文件、worker 日志、`summary.json` 和 `summary.csv`。只有两个阶段的结果都有效时，续评才会跳过该任务；使用相同的 `EVALUATION.output_dir` 时间戳部分即可继续。

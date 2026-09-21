# RoboDojo 评测指南

[English](robodojo.md) | [Benchmark 索引](README_zh.md) | [数据与训练指南](../data/robodojo_zh.md) | [返回项目 README](../../../README_zh.md)

评测器运行 RoboDojo 官方 42 个仿真任务及种子 0、1、2。12 个泛化任务每个种子分别评测标准和 `_random` 配置各 25 局；其他任务每个种子评测 50 局。

## 安装

克隆官方仓库和子模块，并按[官方安装说明](https://github.com/RoboDojo-Benchmark/RoboDojo)准备 Isaac Sim 环境与资源：

```bash
git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git third_party/RoboDojo
(cd third_party/RoboDojo && bash scripts/install.sh)
(cd third_party/RoboDojo && bash scripts/init_assets.sh)
(cd third_party/RoboDojo && bash scripts/robodojo.sh doctor)
```

在 EasyWAM Python 环境启动 manager。仿真客户端默认使用单独的 `RoboDojo` conda 环境。首次运行时，manager 会将轻量的 `easywam_policy` 客户端适配文件复制到 RoboDojo checkout；若目标已有不同内容，则报错而不覆盖。

## 评测

运行完整测评：

```bash
python experiments/robodojo/run_robodojo_manager.py \
  task=robodojo_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=./data/robodojo-lerobot-v3.0/dataset_stats.json
```

只生成所选任务与种子清单、不加载 checkpoint 或启动 Isaac Sim，可设置 `MULTIRUN.create_only=true`。manager 默认将全部所选任务写入输出目录的 `tasks.jsonl`，也可用 `MULTIRUN.task_file=<path>` 指定路径；正式运行时会根据已有结果另生成仅包含待评任务的 worker 清单。

```bash
python experiments/robodojo/run_robodojo_manager.py \
  EVALUATION.task_name=cover_blocks 'EVALUATION.seeds=[0]' \
  MULTIRUN.create_only=true
```

默认在 8 张 GPU 上各加载一份模型，并各运行一个 Isaac Sim 进程。每个仿真进程绑定指定的物理 GPU，进程内使用 CUDA 设备 0。每个仿真进程使用上游配置的环境数（通常为 10；`_random` 配置由 RoboDojo 限制上限）。XPolicyLab 适配器开启进程内批量环境，同一模型 worker 也会动态合并多个客户端的推理请求。可根据显存调整 `MULTIRUN.inference_batch_size`、`MULTIRUN.inference_batch_wait_ms` 和 `EVALUATION.sim_num_envs`。

模型与仿真使用不同 GPU 池：

```bash
python experiments/robodojo/run_robodojo_manager.py \
  task=robodojo_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  MULTIRUN.policy_num_gpus=2 'MULTIRUN.policy_gpu_ids=[0,1]' \
  MULTIRUN.env_num_gpus=4 'MULTIRUN.env_gpu_ids=[2,3,4,5]'
```

单任务单局连通性检查：

```bash
python experiments/robodojo/run_robodojo_manager.py \
  task=robodojo_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.task_name=cover_blocks 'EVALUATION.seeds=[0]' \
  EVALUATION.eval_num_episodes=1 MULTIRUN.num_gpus=1
```

官方 `_result.json` 和三相机视频保存在 `third_party/RoboDojo/eval_result/RoboDojo/`；EasyWAM 的日志、实际配置、`summary.json` 与 `summary.csv` 保存在 `evaluate_results/robodojo/<task>/<timestamp>/`。汇总中的 `success_rate` 和 `score` 使用与官方报告一致的百分比（0–100）。完整正式测评还会更新 RoboDojo 的 `_summary.md`。复用 `EVALUATION.output_dir=<existing-directory>` 即可跳过已有完整结果并续评。只有当前 Python 环境已安装 RoboDojo 和 Isaac Sim 时才设置 `EVALUATION.sim_env=null`。

# RoboCasa365 评测指南

[English](robocasa.md) | [Benchmark 索引](README_zh.md) | [数据与训练指南](../data/robocasa_zh.md) | [返回项目 README](../../../README_zh.md)

评测器实现官方 Human300 多任务协议：atomic_seen 18 个任务、composite_seen 16 个任务、composite_unseen 16 个任务；每个任务运行 50 个 episode，并读取 RoboCasa v1.0.1 或更新版本提供的任务 horizon。

## 在 EasyWAM 环境中安装

在 EasyWAM 当前 Python 环境安装 robosuite master 和官方 RoboCasa。robosuite 虽然已发布到 PyPI，但 RoboCasa 明确要求更新的 master 分支；可以直接通过 pip 安装该分支，不需要保留本地 robosuite clone：

```bash
pip install "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@master"

git clone https://github.com/robocasa/robocasa.git third_party/robocasa
pip install gymnasium pygame h5py lxml hidapi "tianshou==0.4.10"
pip install --no-deps -e third_party/robocasa

python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

使用 no-dependencies 是有意的：RoboCasa 包内固定的 NumPy 和 LeRobot 版本会与 EasyWAM 训练环境冲突。EasyWAM 已包含 imageio、imageio-ffmpeg、termcolor、Pillow、tqdm、NumPy 和 PyYAML；robosuite 会安装 OpenCV、pynput、SciPy、Numba 等基础依赖，因此上面的命令只保留其余 RoboCasa 运行依赖。请保留 EasyWAM 当前的 NumPy、MuJoCo、datasets 和 LeRobot 兼容栈。同一环境里的 robosuite/MuJoCo 变化也可能影响已有 LIBERO 安装。

## 运行评测

在 pretrain scene/object split 上运行完整 50 个任务：

```bash
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 \
  ckpt=./runs/robocasa_easywam_mot_wan22/<run-id>/checkpoints/weights/<checkpoint>.pt \
  EVALUATION.dataset_stats_path=./runs/robocasa_easywam_mot_wan22/<run-id>/dataset_stats.json
```

默认使用 8 张 GPU、每张卡两个模型 worker、每个 worker 20 个 rollout actor，动态推理 batch 上限 16、等待窗口 5 ms；模型预测 32 个 action，每执行 16 步重新规划。这组默认值资源需求较高；显存不足时应同时降低 `workers_per_gpu`、`env_num_per_worker` 和 `inference_batch_size`。`MULTIRUN.gpu_ids=null` 时使用前 `num_gpus` 张卡；否则从有序的 `gpu_ids` 候选池中选择前 `num_gpus` 项。worker 在选中的 GPU 间轮询分配；每个 worker 独立加载一份模型并持有自己的动态推理 batcher，因此提高 `workers_per_gpu` 也会增加显存占用。

常用 override：

```bash
# 4 张 GPU 仅评测 atomic_seen
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 ckpt=<checkpoint> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  'MULTIRUN.task_sets=[atomic_seen]' MULTIRUN.num_gpus=4 \
  'MULTIRUN.gpu_ids=[1,3,5,7]' \
  MULTIRUN.workers_per_gpu=2 MULTIRUN.env_num_per_worker=4 \
  MULTIRUN.inference_batch_size=4

# 只评测一个任务
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 ckpt=<checkpoint> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  EVALUATION.task_name=CloseFridge

# 切换到 target scene/object split
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 ckpt=<checkpoint> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  EVALUATION.split=target
```

设置 **EVALUATION.video_mode=failures** 或 **all** 可保存三相机 rollout 视频；默认关闭视频。

结果保存在 **evaluate_results/robocasa/robocasa_easywam_mot_wan22/<timestamp>/**。每个任务包含 result JSON 和可选视频；run 根目录包含 worker 日志、解析后的配置、模拟器版本/commit、**summary.json**、**summary.csv** 与 **task_success_rates.csv**。worker 失败时也会先写入部分汇总再报错。overall 按所有 episode 加权；当每个任务均为 50 次时，它等价于官方 50-task leaderboard 平均。

复用同一个输出目录即可断点续跑：

```bash
python experiments/robocasa/run_robocasa_manager.py \
  task=robocasa_easywam_mot_wan22 ckpt=<checkpoint> \
  EVALUATION.dataset_stats_path=<dataset_stats.json> \
  EVALUATION.output_dir=<existing-output-directory>
```

只有 task set、任务名、split、episode 数均匹配，且成功/失败 episode 列表完整的结果才会被跳过。

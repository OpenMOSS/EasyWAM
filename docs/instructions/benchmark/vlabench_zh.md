# VLABench 评测指南

[English](vlabench.md) | [Benchmark 索引](README_zh.md) | [数据与训练](../data/vlabench_zh.md)

评测入口为 `experiments/vlabench/run_vlabench_manager.py`，配置为 `configs/benchmark/sim_vlabench.yaml`。所有命令从项目根目录执行。

## 安装

在现有环境中安装以下额外依赖。NumPy、h5py、OpenCV、imageio、PyYAML 和 msgpack 已由项目及其依赖提供，不再重复列出。SciPy 和 scikit-learn 随 Open3D 自动安装；`openai` 是上游任务注册时导入的依赖，评测不会调用其 API。

克隆[官方 VLABench 仓库](https://github.com/OpenMOSS/VLABench)并固定到适配版本，再安装仿真包、下载官方资产。下面的 `scripts/download_assets.py` 来自所克隆的官方仓库：

```bash
git clone https://github.com/OpenMOSS/VLABench.git third_party/VLABench
git -C third_party/VLABench checkout cf588fe60c0c7282174fe979f5913170cfe69017

pip install mujoco==3.2.2 dm-control==1.0.22 open3d==0.18.0 \
  mediapy==1.2.0 'colorlog>=6,<7' 'openai>=1,<3' \
  gym==0.26.2 ipdb gdown 'websockets>=14,<17' \
  'rrt-algorithms @ git+https://github.com/motion-planning/rrt-algorithms.git@14e47963780affe3ff3bb9fc33a4b14e1a733e85'
pip install --no-deps -e third_party/VLABench
MUJOCO_GL=osmesa python third_party/VLABench/scripts/download_assets.py

```

默认上游目录为 `third_party/VLABench`，改用其他目录时同步传入 `EVALUATION.vlabench_root`。模型与仿真均使用当前环境的 Python，无需设置 `EVALUATION.sim_python`。

默认使用 OSMesa 软件渲染，Linux 需安装系统 OSMesa 库；Ubuntu/Debian 可执行 `sudo apt-get install -y libosmesa6`。仿真资产与训练数据独立，下载需要数 GB 空间。训练数据及统计文件按[数据指南](../data/vlabench_zh.md)从 `OpenMOSS-Team/vlabench-lerobot-v3.0` 下载。

## 配置与运行

先生成官方任务清单，检查上游目录与任务配置；此步骤不需要 checkpoint、训练数据或启动仿真：

```bash
python experiments/vlabench/run_vlabench_manager.py \
  MULTIRUN.create_only=true \
  EVALUATION.output_dir=evaluate_results/vlabench/inventory
```

正式评测前，将 `VLABENCH_CHECKPOINT` 设置为实际训练产出的 `.pt` 权重文件路径，例如在终端执行 `export VLABENCH_CHECKPOINT="实际的权重文件路径"`，再运行：

```bash
python experiments/vlabench/run_vlabench_manager.py \
  task=vlabench_easywam_mot_wan22 \
  ckpt="${VLABENCH_CHECKPOINT}" \
  EVALUATION.dataset_stats_path=./data/vlabench-lerobot-v3.0/dataset_stats.json \
  EVALUATION.output_dir=evaluate_results/vlabench/my_run
```

`task` 必须与训练架构一致，可选项目中的五个 `vlabench_easywam_*_wan22` 配方。示例读取下载数据集的统计；若训练使用了其他统计文件，须将 `EVALUATION.dataset_stats_path` 改为该训练配套文件。

默认使用 1 张 GPU、每 GPU 一个模型进程及两个仿真 actor，动态推理 batch=2。可通过 `MULTIRUN.num_gpus`、`MULTIRUN.gpu_ids`、`MULTIRUN.env_num_per_worker` 和 `MULTIRUN.inference_batch_size` 调整。单任务连通性检查可在正式命令后追加 `'EVALUATION.tracks=[track_2_cross_category]' 'EVALUATION.tasks=[insert_flower]' EVALUATION.num_trials=1`；该检查仍使用真实模型和仿真，不代表完整测评。

## 评测范围与结果

默认选择官方固定 track 1、2、3、4、6，每个任务统一使用清单的前 10 个配置，共 500 个 episode。该版本 track 2 的 `insert_flower` 仅有 10 个配置；沿用官方 evaluator 的前置条件，任何所选任务配置数不足时直接报错，不截断、不重复或跳过。仅选择配置足够的任务时才能设 `EVALUATION.num_trials=50`，比较分数时应报告实际评测数量。自定义 track 5 需显式设置 `EVALUATION.tracks=[track_5_cross_task]` 并提供 `EVALUATION.custom_track_file`。

正式评测保持 `EVALUATION.episode_horizon=null`，使用官方任务步数上限，保留官方绝对 EEF 动作、IK、单次环境步进、终止判断及评分语义。模型服务启动前检查 checkpoint 的键与张量形状是否完整，拒绝缺失策略权重的 checkpoint。

结果写入指定输出目录，包括 `manifest.json`、`tasks.json`、逐 episode JSON、日志、`summary.json` 和 `summary.csv`。SR、PS、IS 分别为成功率、进度分数和意图分数；先按任务平均，再按 track 平均，最后对 track 做宏平均。覆盖不完整时正式指标为 null，同时提供部分指标和覆盖数。

使用相同 `EVALUATION.output_dir` 可续评并跳过完整结果；权重、统计或评测配置改变时应使用新目录。仿真或通信失败最多尝试两次；`EVALUATION.video_mode=failures` 或 `all` 可保存 rollout 视频。

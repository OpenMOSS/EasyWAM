# LIBERO-Pro 评测指南

[English](libero_pro.md) | [Benchmark 索引](README_zh.md) | [LIBERO 数据指南](../data/libero_zh.md) | [返回项目 README](../../../README_zh.md)

LIBERO-Pro 使用在标准 LIBERO suite 上训练得到的 checkpoint，评测 Object、Position、Semantic、Task 和 Environment 五类泛化扰动。默认协议覆盖四个 suite、20 个 suite/扰动组合、200 个任务，每个任务执行 50 次，共 10,000 个 episode。

## 安装

将官方仓库克隆到 EasyWAM 的 `third_party` 目录：

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git \
  third_party/LIBERO-PRO
```

LIBERO-Pro 与 LIBERO 使用相同的仿真依赖。在 EasyWAM 环境中安装评测所需的软件包：

```bash
pip install mujoco easydict robosuite bddl future \
  cloudpickle gym huggingface_hub
```

不要执行 `pip install -e third_party/LIBERO-PRO`。LIBERO-Pro 与标准 LIBERO 使用相同的 Python 包名；专用 manager 会直接加载指定的源码目录，避免覆盖其他 benchmark 使用的 LIBERO 包。

## 下载 BDDL 与初始状态

主榜 BDDL 和初始状态单独发布。按照 LIBERO-Pro 仓库要求，先将数据下载到仓库内的临时目录：

```bash
hf download zhouxueyang/LIBERO-Pro \
  --repo-type dataset \
  --local-dir third_party/LIBERO-PRO/libero_data
```

再将文件移动到 LIBERO-Pro 规定的目录：

```bash
mv third_party/LIBERO-PRO/libero_data/bddl_files/* \
  third_party/LIBERO-PRO/libero/libero/bddl_files/
mv third_party/LIBERO-PRO/libero_data/init_files/* \
  third_party/LIBERO-PRO/libero/libero/init_files/
```

最终目录至少需要包含：

```text
third_party/LIBERO-PRO/
└── libero/libero/
    ├── bddl_files/
    │   ├── libero_goal_object/
    │   ├── libero_goal_swap/
    │   └── ...
    └── init_files/
        ├── libero_goal_object/
        ├── libero_goal_swap/
        └── ...
```

EasyWAM 默认配置与该目录结构对应：

```yaml
EVALUATION:
  libero_pro_root: ./third_party/LIBERO-PRO
  libero_pro_data_root: ${EVALUATION.libero_pro_root}/libero/libero
```

manager 从 `libero_pro_root` 加载源码，从 `libero_pro_data_root` 读取 BDDL 和初始状态，并在评测输出目录生成隔离的 LIBERO 路径配置，不会改写上游仓库。如果仓库克隆在其他位置，只需覆盖 `EVALUATION.libero_pro_root`；除非显式覆盖，数据路径会自动跟随源码目录。

## 校验与评测

先在不加载模型和 GPU 的情况下校验源码、assets、20 个 suite 和全部任务文件：

```bash
python experiments/libero_pro/run_libero_pro_manager.py \
  MULTIRUN.create_only=true
```

完整官方评测：

```bash
python experiments/libero_pro/run_libero_pro_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  MULTIRUN.num_gpus=4
```

例如仓库位于其他位置时：

```bash
python experiments/libero_pro/run_libero_pro_manager.py \
  MULTIRUN.create_only=true \
  EVALUATION.libero_pro_root=/path/to/LIBERO-PRO
```

开发阶段可以筛选基础 suite、扰动和 task ID：

```bash
python experiments/libero_pro/run_libero_pro_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial]' \
  'MULTIRUN.perturbations=[object,semantic]' \
  'MULTIRUN.task_ids=[0]'
```

有效扰动名称为 `object`、`position`、`semantic`、`task` 和 `environment`。task ID 从 0 开始，并应用到每个选中的 suite/扰动组合。默认使用官方步数预算：Spatial 220、Object 280、Goal 300、LIBERO-10 520；每个任务必须提供至少 50 个初始状态。

## 并发、结果与续评

`MULTIRUN.num_gpus`、`gpu_ids`、`workers_per_gpu`、`env_num_per_worker`、`inference_batch_size` 和 `inference_batch_wait_ms` 与其他 EasyWAM benchmark 含义一致。每个 model worker 独立加载模型，worker 内的 rollout actor 共享动态推理 batcher。

结果写入 `evaluate_results/libero_pro/<task>/<timestamp>/`，包括任务清单、上游源码信息、隔离的运行时配置、worker 日志、逐任务结果、错误记录、`summary.json`、`summary.csv`、`task_results.csv` 和 `leaderboard.csv`。`leaderboard.csv` 使用 0–1 成功率，按四个 suite × 五类扰动输出官方矩阵和宏平均。

重新使用同一个 `EVALUATION.output_dir` 会跳过协议元数据和 50 个 episode 均完整的任务；损坏或 trial 数不足的结果会重新评测。部分运行会明确报告 coverage，只有所有选中单元格完整时才标记为完整 leaderboard。

本入口只覆盖官方五维单扰动主榜，不生成多扰动组合，也不包含 Hugging Face 数据集中的后续七类扩展。

# LIBERO-Plus 使用指南

[English](libero_plus.md) | [Benchmark 索引](README_zh.md) | [LIBERO 数据指南](../data/libero_zh.md) | [返回项目 README](../../../README_zh.md)

LIBERO-Plus 是鲁棒性评测 benchmark。EasyWAM 直接评测在 LIBERO 上训练得到的 checkpoint，不需要单独的 LIBERO-Plus 训练 task。

## 独立环境

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) 与原版 LIBERO 使用相同的 `libero` Python 包名，因此请为 LIBERO-Plus 使用独立环境，或者在现有环境中替换原版包。

安装 EasyWAM 后，按照上游说明安装 fork 及其额外的系统和 Python 依赖。典型流程如下：

```bash
sudo apt install libexpat1 libfontconfig1-dev libmagickwand-dev
pip install wand scikit-image
git clone https://github.com/sylvestf/LIBERO-plus.git <path/to/LIBERO-plus>
pip install --no-deps -e <path/to/LIBERO-plus>
```

从[官方 LIBERO-Plus 资源页面](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main)下载 `assets.zip`，并解压到 fork 的 `libero/libero/assets/` 目录。

`${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml` 必须将 `benchmark_root`、`bddl_files`、`init_states` 和 `assets` 指向 LIBERO-Plus 目录。manager 会在启动 worker 前检查这些路径、扩展资源目录、`task_classification.json`、BDDL 文件以及官方 suite 规模。

## Checkpoint 与统计文件

可以使用任意通过 LIBERO task 训练的 EasyWAM checkpoint，例如：

- `libero_easywam_mot_wan22` 或 `libero_easywam_mot_wan22_lora`
- `libero_easywam_unified_wan22` 或 `libero_easywam_unified_wan22_lora`
- `libero_easywam_hidden_wan22` 或 `libero_easywam_hidden_wan22_lora`

评测必须提供 `dataset_stats.json`。如果没有设置 `EVALUATION.dataset_stats_path`，manager 会在 checkpoint 向上的四层父目录中自动查找。

## 完整评测

官方配置对每个选中的扰动任务评测一次，四个 suite 共包含 10,030 个任务。

```bash
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  MULTIRUN.num_gpus=3 'MULTIRUN.gpu_ids=[0,2,4,6]' \
  MULTIRUN.workers_per_gpu=2 MULTIRUN.env_num_per_worker=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10
```

开发阶段可以按照 suite、扰动类别、难度或从零开始的 task ID 进行筛选：

```bash
python experiments/libero_plus/run_libero_plus_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial]' \
  'MULTIRUN.categories=[camera,light]' \
  'MULTIRUN.difficulty_levels=[1,2]' \
  'MULTIRUN.task_ids=[0,1,2]'
```

有效类别为 `layout`、`camera`、`robot`、`language`、`light`、`background` 和 `noise`；难度范围为 1–5。task ID 会应用到每一个选中的 suite。

`MULTIRUN.gpu_ids=null` 时 manager 使用前 `num_gpus` 张卡；否则从有序的 `gpu_ids` 候选池中选择前 `num_gpus` 项。模型 worker 在选中的 GPU 间轮询分配。每个 worker 独立加载一份模型，其 rollout actor 动态领取任务并共享该 worker 的推理组批器；提高 `workers_per_gpu` 会增加模型显存占用。EGL 或 OSMesa 根据每张 GPU 上实际激活的环境总并发数选择。默认关闭视频和进度输出。

结果写入 `evaluate_results/libero_plus/<task>/<timestamp>/`，包括 `tasks.jsonl`、worker 日志、逐任务结果、错误记录、`summary.json`、`summary.csv` 和 `task_results.csv`，并按 suite、扰动类别、难度以及类别与难度组合汇总。重新使用同一个 `EVALUATION.output_dir` 时会跳过有效结果，从中断位置继续。

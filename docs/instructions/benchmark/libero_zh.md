# LIBERO 评测指南

[English](libero.md) | [Benchmark 索引](README_zh.md) | [数据与训练指南](../data/libero_zh.md) | [返回项目 README](../../../README_zh.md)

## 安装

在 EasyWAM 所在的 Python 3.10 环境中安装 LIBERO：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO

# 这里只安装 LIBERO 评测需要、但 EasyWAM 未声明的包。
pip install mujoco==3.3.2 easydict==1.9 \
  robosuite==1.4.0 bddl==1.0.1 future==0.18.2 \
  cloudpickle==2.1.0 gym==0.25.2

# LIBERO 的 setup.py 没有运行时依赖，避免重复安装或降级 EasyWAM 的包。
pip install --no-deps -e third_party/LIBERO
```

首次导入时，LIBERO 会询问 demonstration dataset 的保存位置。评测只使用源码中自带的 BDDL、assets 和初始状态，因此选择默认路径即可：

```bash
printf 'n\n' | python -c "import libero.libero"
```

这会生成 `~/.libero/config.yaml`。如果设置了 `LIBERO_CONFIG_PATH`，配置会写到该目录；运行评测时需要保持这个环境变量一致。官方依赖中的 `robomimic` 和 `thop` 只服务于 LIBERO 自带的策略训练，EasyWAM 评测不会使用，因此不安装。Hydra、NumPy、Weights & Biases、Transformers、Einops、PyTorch、Pillow、Termcolor 和 tqdm 已由 EasyWAM 提供。

上游源码及版本清单见官方 [LIBERO 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)。

## 评测

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
# 只评测部分 suite，每张 GPU 启动两个模型 worker，每个 worker 启动四个 actor
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object]' \
  MULTIRUN.num_gpus=3 'MULTIRUN.gpu_ids=[1,3,5,7]' \
  MULTIRUN.workers_per_gpu=2 \
  MULTIRUN.env_num_per_worker=4 \
  MULTIRUN.inference_batch_size=4 MULTIRUN.inference_batch_wait_ms=10
```

默认协议会评测四个 suite，每个任务执行 50 次。`MULTIRUN.gpu_ids=null` 时使用前 `num_gpus` 张卡；否则从有序的 `gpu_ids` 候选池中选择前 `num_gpus` 项。模型 worker 在选中的 GPU 间轮询分配；每个 worker 独立加载一份模型，其 rollout actor 动态领取任务并共享该 worker 的推理组批器。提高 `workers_per_gpu` 会增加模型显存占用。同一 GPU 只有一个活动环境时使用 EGL，存在并发环境时使用 OSMesa。默认关闭视频和进度渲染，可通过 `EVALUATION.video_mode`、`EVALUATION.visualize_future_video` 和 `EVALUATION.progress` 调整。

结果保存在 `evaluate_results/libero/<task>/<timestamp>/`，其中包括 worker 日志、逐任务 JSON、`summary.json`、`summary.csv` 和 `task_success_rates.csv`。重新指定同一个 `EVALUATION.output_dir` 即可续评，manager 会跳过结果完整的任务。

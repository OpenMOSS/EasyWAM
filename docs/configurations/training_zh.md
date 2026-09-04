# 数据、训练与评测配置

[English](training.md) | [配置索引](README_zh.md) | [返回 README](../../README_zh.md)

本文说明 Hydra 完整配置中模型以外的部分。默认值来自 `configs/train.yaml`，数据和 task recipe 会针对具体 Benchmark 覆盖其中一部分。

## 数据配置

`configs/data/libero_2cam.yaml` 和 `configs/data/robotwin.yaml` 会实例化 `RobotVideoDataset` 与 `WAMProcessor`。添加或修改数据时应保持以下配置组一致：

| 配置组 | 重要设置 |
| --- | --- |
| 数据位置 | `dataset_dirs`、`text_embedding_cache_dir`、可选的 `pretrained_norm_stats` |
| 观测形状 | `shape_meta.images`、`video_size`、`concat_multi_camera`、`num_output_cameras` |
| 序列采样 | `num_frames`、`global_sample_stride`、`action_video_freq_ratio` |
| 机器人张量 | `shape_meta.action`、`shape_meta.state`、`action_output_dim`、`proprio_output_dim` |
| 归一化 | `norm_default_mode`、`norm_exception_mode`、`use_stepwise_action_norm`、可选 transform |
| 数据划分 | `val_set_proportion`、`is_training_set`、`skip_padding_as_possible` |

`num_frames` 是 action/state 序列长度，`action_video_freq_ratio` 从该序列稀疏选择视频时间点。例如，仓库中的 33 步数据配合比例 `4`，对应 32 个未来动作和包括观测帧在内的 9 帧视频。修改任意一项都会改变训练张量形状，并且必须与评测 horizon 兼容。

每个图像条目包含源 `raw_shape` 和 transform 后的 `shape`。Resize transform、`shape`、最终 `video_size`、相机拼接方式和 `num_output_cameras` 必须描述同一个布局。LIBERO 横向拼接两个 224×224 相机；RoboTwin 使用三相机组合方式。

Action 和 proprio 维度通过 Hydra 插值传给模型。如果 processor 改变了维度或动作表示，需要同步更新 `shape_meta` 与对应的 processor 输出维度。

### 归一化统计

- LIBERO 默认使用 `min/max` 归一化，并可从训练数据计算统计量。
- RoboTwin 默认使用 `z-score`，通过 `pretrained_norm_stats` 加载已有数据集统计量。
- 评测必须使用训练 checkpoint 对应的统计量；无法自动找到时应传入 `EVALUATION.dataset_stats_path`。

### 文本 Embedding

训练通常设置 `model.backbone.load_text_encoder=false`，并从 `text_embedding_cache_dir` 读取 embedding。应使用与训练相同的 task/backbone 选择生成 cache：

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
```

使用相同 `text_encoder_id` 和兼容 context 长度的模型可以共享 cache。即使 prompt 文本相同，也不要跨不同 encoder 复用 embedding。

## 训练配置

| 设置 | 默认值 | 行为 |
| --- | ---: | --- |
| `batch_size` | `16` | 梯度累积前，每个进程的 DataLoader batch size；task recipe 可能覆盖。 |
| `gradient_accumulation_steps` | `1` | 优化器累积步数；有效全局 batch 为单进程 batch × 进程数 × 累积步数。 |
| `learning_rate` | `1e-4` | AdamW 学习率。 |
| `weight_decay` | `0.0` | AdamW weight decay；仓库 task 通常会覆盖。 |
| `lr_scheduler_type` | `cosine` | Trainer 学习率 schedule。 |
| `warmup_ratio` | `0.05` | Warmup 步数比例，必须位于 `[0, 1)`。 |
| `max_steps` | `1000` | 必填正数的优化步数上限；项目不使用基于 epoch 的训练。 |
| `mixed_precision` | `bf16` | 可选值为 `no`、`fp16` 或 `bf16`。 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值。 |
| `seed` | `42` | 训练和 DataLoader worker 随机种子。 |
| `resume` | `null` | 恢复训练输入；新训练保持 null。 |

Launcher 控制分布式执行，Hydra 配置控制训练行为：

```bash
# 单机 8 进程，ZeRO-1
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot_wan22

# 双机，每台 8 进程，ZeRO-2
NNODES=2 NODE_RANK=0 MASTER_ADDR=<host> MASTER_PORT=29500 \
  NPROC_PER_NODE=8 bash scripts/train_zero2.sh task=robotwin_easywam_mot_wan22
```

第二个节点应设置 `NODE_RANK=1`。所有节点必须使用相同的 `NNODES`、地址、端口、task 和 override。除非在命令行后续再次覆盖，训练脚本会根据选中的 task 设置输出目录和 WandB 名称。

## 日志、Checkpoint 与训练时评测

| 设置 | 用途 |
| --- | --- |
| `output_dir` | 当前运行的 checkpoint、日志和评测产物目录。 |
| `log_every` | 训练指标记录间隔。 |
| `save_every` | Checkpoint 保存间隔。 |
| `eval_every` | 存在验证集时，验证推理的步数间隔。 |
| `eval_num_inference_steps` | 训练时评测采用的去噪步数。 |
| `eval_save_video` | 每个 rank 在评测时保存一份拼接后的预测/VAE/真值视频。 |
| `wandb.*` | 启用 WandB 并配置 workspace、project、运行名称、group 和 mode。 |

各间隔可根据 `max_steps` 设为正数，也可设为 `0` 以关闭对应行为。保存视频有助于定性检查，但会增加解码、同步和存储开销。

## 评测配置

评测根配置是 `configs/sim_libero.yaml`、`configs/sim_libero_plus.yaml` 和 `configs/sim_robotwin.yaml`。它们继承 `train.yaml`，选择 task，启用运行时 text encoder，跳过重复的基础 DiT 初始化，并从 `ckpt` 加载训练权重。

通用 policy 设置包括：

| 设置 | 含义 |
| --- | --- |
| `EVALUATION.action_horizon` | 预测动作数量；`null` 表示从数据序列配置推导。 |
| `EVALUATION.replan_steps` | 请求下一次预测前最多执行的动作数。 |
| `EVALUATION.num_inference_steps` | 视频/动作推理的去噪步数。 |
| `EVALUATION.sigma_shift` | 可选的推理 scheduler shift override。 |
| `EVALUATION.text_cfg_scale` / `negative_prompt` | 文本 classifier-free guidance 输入。 |
| `EVALUATION.dataset_stats_path` | 与 checkpoint 配套的归一化统计。 |
| `EVALUATION.device` / `rand_device` | 模型执行和初始随机噪声所在设备。 |
| `EVALUATION.output_dir` | 结果、日志、汇总与 rollout 输出目录。 |

运行时会将 `replan_steps` 限制在预测 action horizon 内。缩短 replanning 间隔可以更频繁地响应环境，但会更频繁地调用模型。`num_inference_steps` 必须为正数；减少步数会降低去噪计算量，但可能损失 policy 质量。

`MULTIRUN.num_gpus` 和 `MULTIRUN.max_tasks_per_gpu` 控制 manager 的任务分片，不是训练 world size。Benchmark 专属筛选条件和断点续评规则请参阅 [LIBERO](../instructions/libero_zh.md)、[LIBERO-Plus](../instructions/libero_plus_zh.md) 和 [RoboTwin](../instructions/robotwin_zh.md) 指南。

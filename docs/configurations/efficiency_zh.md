# 效率配置

[English](efficiency.md) | [配置索引](README_zh.md) | [返回 README](../../README_zh.md)

EasyWAM 分别提供输入吞吐、模型显存、Attention kernel 和闭环评测相关的控制项。应一次调整一组，并在 warmup 后测量稳态吞吐；最佳值取决于 GPU 显存、CPU 核数、存储、序列形状和模型架构。

## 数据加载

| 设置 | 默认值 | 效果与权衡 |
| --- | ---: | --- |
| `num_workers` | `8` | 增加 worker 可以隐藏解码和 transform 延迟，但会消耗 CPU 与主机内存。设为 `0` 可用于进程内加载和调试。 |
| `dataloader_prefetch_factor` | `16` | 每个 worker 预取的 batch 数。增大可缓解 I/O 抖动，但会消耗更多主机内存；仅在启用 worker 时生效。 |
| `dataloader_persistent_workers` | `true` | 在 DataLoader 迭代间保留 worker，避免重复创建进程；仅在启用 worker 时生效。 |
| `dataloader_pin_memory` | `true` | 使用锁页内存改善 CUDA 传输吞吐，同时增加 pinned memory 占用。 |
| `dataloader_worker_threads` | `1` | 设置每个 worker 内的 PyTorch CPU 线程数，避免 worker 层面的线程过量。 |

可以逐步增加 `num_workers`，直到 accelerator 利用率不再改善，再调整 prefetch。共享机器上的 CPU 压力取决于进程数、每进程 worker 数和每 worker 线程数三者的乘积，而不是其中某个值。

## 训练显存与分布式执行

| 控制项 | 吞吐/显存行为 |
| --- | --- |
| `mixed_precision=bf16` | 默认 GPU 训练模式，降低 activation 占用和张量带宽需求，且不需要 FP16 loss scaling；硬件必须支持 BF16。 |
| `model.backbone.use_gradient_checkpointing=true` | 反向传播时重算 Transformer activation 以节省显存，训练速度会下降；同时作用于 video 和已配置的 action expert。 |
| `_lora` task recipe | 减少可训练参数和优化器状态；activation 显存仍取决于 batch 和序列形状。 |
| `scripts/train_zero1.sh` | 使用 DeepSpeed ZeRO-1 切分优化器状态。 |
| `scripts/train_zero2.sh` | 使用 ZeRO-2 进一步切分梯度，通常节省更多设备显存。 |
| `scripts/train_zero2_offload.sh` | 将 ZeRO-2 优化器状态卸载到 CPU；适合设备显存受限场景，但有 PCIe/CPU 开销。 |
| `gradient_accumulation_steps` | 不增加单个 micro-batch 的情况下提高有效 batch，但每个优化器 step 需要更多前向和反向计算。 |

OOM 时应先减小 `batch_size`，然后依次考虑 gradient checkpointing、ZeRO-2、在实验目标允许时使用 LoRA，最后再使用以显存为优先的 CPU Offload。

## Attention Backend

`model.backbone.attention_backend` 可设为 `auto`、`fa4`、`fa3`、`fa2` 或 `sdpa`。

- `auto` 按 FA4、FA3、FA2、PyTorch SDPA 的顺序选择当前已安装且适用于 CUDA dtype 和 head 维度的 kernel。
- 显式指定 FlashAttention backend 时必须安装对应包；无法加载其 `flash_attn_func` API 会提前报错。
- FlashAttention kernel 要求合适的 CUDA FP16/BF16 张量和受支持的 head 维度。对于设备、dtype 或 mask 布局不支持的单次调用，会回退到 SDPA。
- `sdpa` 是兼容性基线，不需要可选 FlashAttention 包。

常规运行建议使用 `auto`，排查 kernel 兼容问题时使用 `sdpa`。应查看启动日志，确认每种 Attention 布局最终选择的 backend。

## VAE Batching 与推理 Cache

| 设置 | 默认值 | 效果与权衡 |
| --- | ---: | --- |
| `vae_micro_batch_size` | `null` | `null` 表示整 batch 处理以获得最大 batching；正整数表示分块执行 VAE，降低峰值显存。`1` 显存最低、batching 最弱。 |
| `inference_cross_kv_reuse` | `true` | 在一次推理调用内复用静态 cross-attention 投影。兼容性排查或 cache 等价性测试时可关闭。 |

VAE micro-batching 会同时应用于训练和评测的模型构造，但不会改变数学意义上的 batch 或优化器 batch size。Cross-K/V 复用只用于推理，不会跨环境 replan 保留。

## 推理与评测延迟

| 设置 | 效果与权衡 |
| --- | --- |
| `EVALUATION.num_inference_steps` | 主要去噪计算倍数。减少步数可降低延迟，但可能降低预测质量。 |
| `EVALUATION.action_horizon` | 预测动作数。更长的 chunk 可以摊薄模型调用，但增加 action token 计算并依赖更长的开环预测。 |
| `EVALUATION.replan_steps` | 每次预测后执行的动作数。值越小，模型调用越频繁；运行时会将其限制在 action horizon 内。 |
| `EVALUATION.torch_compile` | 编译各架构声明的张量密集推理函数。首次调用编译可能较慢，适合之后重复使用稳定 shape。 |
| `EVALUATION.torch_compile_mode` | 默认 `reduce-overhead`，适合重复的 batch-1 推理，条件允许时可能使用 CUDA Graph。 |
| `EVALUATION.skip_get_obs_within_replan` | RoboTwin 专属优化，在两次 replan 之间跳过 RGB 渲染；保存的视频会因此缺少中间渲染帧。 |
| `EVALUATION.video_mode`、`visualize_future_video`、`eval_save_video` | 视频编码、解码、可视化和磁盘写入都会增加开销；吞吐测试应保持关闭。 |
| `MULTIRUN.num_gpus`、`max_tasks_per_gpu` | 控制评测任务并发；worker 过多可能争用 GPU 显存和渲染资源。 |

`torch_compile_backend`、`torch_compile_fullgraph`、`torch_compile_dynamic` 和 `torch_compile_options` 会传给 `torch.compile`。应先使用仓库默认值。已经加载的模型不允许切换到另一套 compile 配置；修改这些设置后需要重启 worker。

## 起始配置方案

以下方案仅用于开始调优，不是通用 Benchmark 设置。

### 吞吐优先

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_wan22 \
  mixed_precision=bf16 \
  model.backbone.attention_backend=auto \
  model.backbone.use_gradient_checkpointing=false \
  vae_micro_batch_size=null
```

先使用能够稳定运行的最大单进程 batch，再根据 accelerator 空闲时间的测量结果调整 DataLoader worker 和 prefetch。

### 显存优先

```bash
NPROC_PER_NODE=8 bash scripts/train_zero2_offload.sh \
  task=libero_easywam_mot_wan22_lora \
  batch_size=1 \
  gradient_accumulation_steps=8 \
  model.backbone.use_gradient_checkpointing=true \
  vae_micro_batch_size=1
```

CPU Offload 和 checkpointing 都会用速度换容量。如果其他调整后模型已经能够放入显存，应优先移除 Offload。

### 兼容性与问题排查

```bash
python scripts/train.py --cfg job \
  task=libero_easywam_mot_wan22 \
  model.backbone.attention_backend=sdpa \
  num_workers=0 \
  vae_micro_batch_size=1 \
  inference_cross_kv_reuse=false
```

实际诊断运行时，应把相同 override 传给所需 launcher 或 evaluator。该方案优先保证执行行为可预测、问题容易隔离，而不是性能。

## 测量检查清单

- 对比时保持模型、数据、全局 batch、序列形状、精度和推理步数一致。
- 稳态计时应排除文本 cache 生成、模型加载、编译 warmup 和首次显存分配。
- 同时观察 accelerator 利用率、设备峰值显存、主机内存、CPU 饱和度和存储吞吐。
- 每次只调整一组配置，并将完整 Hydra 组合配置与结果一同保存。

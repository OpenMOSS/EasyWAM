# Efficiency Configuration

[中文](efficiency_zh.md) | [Configuration index](README.md) | [Back to README](../../README.md)

EasyWAM exposes independent controls for input throughput, model memory, attention kernels, and closed-loop evaluation. Tune one group at a time and measure steady-state throughput after warmup; the best values depend on GPU memory, CPU cores, storage, sequence shape, and model architecture.

## Data loading

| Setting | Default | Effect and tradeoff |
| --- | ---: | --- |
| `num_workers` | `8` | More workers can hide decoding/transform latency, but consume CPU and host memory. Set `0` for in-process loading and debugging. |
| `dataloader_prefetch_factor` | `16` | Batches queued per worker. Larger values smooth I/O stalls at higher host-memory cost. Used only when workers are enabled. |
| `dataloader_persistent_workers` | `true` | Keeps workers alive between DataLoader iterations, avoiding process startup cost. Used only when workers are enabled. |
| `dataloader_pin_memory` | `true` | Uses page-locked host tensors to improve CUDA transfer throughput, at higher pinned-memory usage. |
| `dataloader_worker_threads` | `1` | Sets PyTorch CPU threads inside each worker and prevents worker-level oversubscription. |

Increase `num_workers` until accelerator utilization stops improving, then tune prefetching. On shared machines, the product of process count, workers per process, and worker threads is the relevant CPU pressure—not any one value alone.

## Training memory and distributed execution

| Control | Throughput/memory behavior |
| --- | --- |
| `mixed_precision=bf16` | Default GPU training mode; reduces activation and tensor bandwidth requirements without FP16 loss scaling. Requires BF16-capable hardware. |
| `model.backbone.use_gradient_checkpointing=true` | Recomputes transformer activations during backward to save memory; expect slower training. It applies to video and configured action experts. |
| `_lora` task recipes | Reduce trainable parameters and optimizer state. Activation memory still depends on batch and sequence shape. |
| `scripts/train_zero1.sh` | Partitions optimizer state with DeepSpeed ZeRO-1. |
| `scripts/train_zero2.sh` | Also partitions gradients with ZeRO-2, usually saving more device memory. |
| `scripts/train_zero2_offload.sh` | Moves ZeRO-2 optimizer state to CPU; use when device memory is the limiting factor and accept PCIe/CPU overhead. |
| `gradient_accumulation_steps` | Raises effective batch size without increasing one micro-batch, but adds forward/backward work per optimizer step. |

First reduce `batch_size` if a run is out of memory. Then enable gradient checkpointing, select ZeRO-2, use LoRA where scientifically appropriate, or use CPU offload as the final memory-oriented option.

## Attention backend

Set `model.backbone.attention_backend` to one of `auto`, `fa4`, `fa3`, `fa2`, or `sdpa`.

- `auto` tries FA4, FA3, FA2, then PyTorch SDPA, choosing the first installed kernel eligible for the current CUDA dtype and head dimension.
- An explicit FlashAttention backend requires its package and fails early if its `flash_attn_func` API cannot be loaded.
- FlashAttention kernels require eligible CUDA FP16/BF16 tensors with supported head dimensions. Individual calls with unsupported devices, dtypes, or mask layouts fall back to SDPA.
- `sdpa` is the compatibility baseline and requires no optional FlashAttention package.

Use `auto` for normal runs and `sdpa` when debugging kernel compatibility. Check startup logs to confirm the backend actually selected for each attention layout.

## VAE batching and inference caches

| Setting | Default | Effect and tradeoff |
| --- | ---: | --- |
| `vae_micro_batch_size` | `null` | `null` processes the full batch for maximum batching; a positive integer chunks VAE work to reduce peak memory. `1` is the lowest-memory, least-batched mode. |
| `inference_cross_kv_reuse` | `true` | Reuses static cross-attention projections inside one inference call. Disable for compatibility diagnosis or cache-equivalence testing. |

VAE micro-batching applies to both training and evaluation model construction. It does not change the mathematical batch or optimizer batch size. Cross-K/V reuse is inference-only and does not persist across environment replans.

## Inference and evaluation latency

| Setting | Effect and tradeoff |
| --- | --- |
| `EVALUATION.num_inference_steps` | Main denoising compute multiplier. Fewer steps reduce latency but may reduce prediction quality. |
| `EVALUATION.action_horizon` | Number of predicted actions. Larger chunks amortize model calls but consume more action-token compute and rely on longer open-loop predictions. |
| `EVALUATION.replan_steps` | Number of actions executed per prediction. Smaller values increase model-call frequency; runtime clips it to the action horizon. |
| `EVALUATION.torch_compile` | Compiles architecture-specific tensor-heavy inference functions. Useful for repeated stable shapes after a potentially expensive first-call compile. |
| `EVALUATION.torch_compile_mode` | Defaults to `reduce-overhead`, which is suitable for repeated batch-1 inference and may use CUDA graphs when eligible. |
| `EVALUATION.skip_get_obs_within_replan` | RoboTwin-only optimization that skips RGB rendering between replans; saved videos then contain fewer rendered frames. |
| `EVALUATION.video_mode`, `visualize_future_video`, `eval_save_video` | Video encoding, decoding, visualization, and disk writes add overhead; leave disabled for throughput measurements. |
| `MULTIRUN.num_gpus`, `max_tasks_per_gpu` | Control evaluation task concurrency. Excess workers can contend for GPU memory and rendering resources. |

`torch_compile_backend`, `torch_compile_fullgraph`, `torch_compile_dynamic`, and `torch_compile_options` are forwarded to `torch.compile`. Keep the checked-in defaults first. Reusing a loaded model with a different compile configuration is rejected; restart the worker when changing compile settings.

## Starting profiles

These are starting points, not universal benchmark settings.

### Throughput-oriented

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_wan22 \
  mixed_precision=bf16 \
  model.backbone.attention_backend=auto \
  model.backbone.use_gradient_checkpointing=false \
  vae_micro_batch_size=null
```

Use the largest stable per-process batch, then tune DataLoader workers and prefetching from measured accelerator idle time.

### Memory-oriented

```bash
NPROC_PER_NODE=8 bash scripts/train_zero2_offload.sh \
  task=libero_easywam_mot_wan22_lora \
  batch_size=1 \
  gradient_accumulation_steps=8 \
  model.backbone.use_gradient_checkpointing=true \
  vae_micro_batch_size=1
```

CPU offload and checkpointing trade speed for capacity. Remove offload first if the model fits after other changes.

### Compatibility and diagnosis

```bash
python scripts/train.py --cfg job \
  task=libero_easywam_mot_wan22 \
  model.backbone.attention_backend=sdpa \
  num_workers=0 \
  vae_micro_batch_size=1 \
  inference_cross_kv_reuse=false
```

For an actual diagnostic run, apply the same overrides to the desired launcher or evaluator. This profile prioritizes predictable execution and isolation rather than performance.

## Measurement checklist

- Compare identical model, data, global batch, sequence shapes, precision, and inference steps.
- Exclude text-cache generation, model loading, compilation warmup, and first-iteration allocation from steady-state timing.
- Monitor accelerator utilization, peak device memory, host memory, CPU saturation, and storage throughput together.
- Change one configuration group at a time and retain the fully composed Hydra config with the result.

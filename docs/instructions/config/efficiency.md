# Efficiency Configuration

[中文](efficiency_zh.md) | [Configuration index](README.md) | [Back to README](../../../README.md)

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
| `EVALUATION.video_mode`, `visualize_future_video`, `eval_save_video` | Video encoding, decoding, visualization, and disk writes add overhead; leave disabled for throughput measurements. |
| `MULTIRUN.num_gpus`, `gpu_ids`, `workers_per_gpu`, `env_num_per_worker` | Control GPUs, model copies per GPU, and rollout actors per model worker. `gpu_ids=null` selects the first `num_gpus` devices; otherwise the first `num_gpus` entries of the ordered `gpu_ids` candidate pool are selected. Each additional worker loads another model copy. |
| `MULTIRUN.inference_batch_size`, `inference_batch_wait_ms` | Control dynamic inference batch size and queue window. |
| `MULTIRUN.prompt_cache_size` | Control the maximum number of prompt embeddings cached by each model worker. |

`torch_compile_backend`, `torch_compile_fullgraph`, `torch_compile_dynamic`, and `torch_compile_options` are forwarded to `torch.compile`. Keep the checked-in defaults first. Reusing a loaded model with a different compile configuration is rejected; restart the worker when changing compile settings.

### Tune evaluation concurrency on the target machine

Use `scripts/tune_eval_concurrency.py` to run a fixed, representative evaluation workload across candidate values. For example:

```bash
python scripts/tune_eval_concurrency.py \
  --benchmark robocasa \
  --env-counts 4,8 \
  --batch-sizes 2,4,8 \
  --wait-ms 0,10 \
  --repeats 1 \
  --timeout-seconds 3600 \
  --monitor-gpu-ids 2,3,5 \
  -- \
  task=robocasa_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json> \
  EVALUATION.num_trials=2 \
  'MULTIRUN.task_sets=[atomic_seen]' \
  MULTIRUN.num_gpus=3 \
  'MULTIRUN.gpu_ids=[2,3,5,6]'
```

The values after `--` are forwarded to the selected benchmark manager. The tuner owns and replaces only `EVALUATION.output_dir`, `env_num_per_worker`, `inference_batch_size`, and `inference_batch_wait_ms`. It skips combinations where the batch size exceeds the actor count, gives every run an isolated output directory, detects nonzero exits, timeouts, and common OOM messages, and samples `nvidia-smi` memory when available.

Results are stored under `evaluate_results/concurrency_tuning/<benchmark>/<timestamp>/` as `results.json`, `summary.csv`, per-run commands, and manager logs. The recommendation is the smallest fully successful configuration within 3% of the fastest median wall time. Use `--dry-run` to inspect commands without launching evaluation.

Start with the small grid above, then rerun the best neighboring values with `--repeats 2` or more before adopting the result.

Use the same tasks, episode count, checkpoint, inference steps, and GPU pool for every candidate. The workload must contain enough simultaneously pending tasks to exercise the largest `env_num_per_worker`; a single task cannot measure actor concurrency. Model startup is included in wall time, so use enough rollouts to make startup a small fraction of the run. Run on otherwise idle GPUs because memory samples include all processes on the selected devices.

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

# Data, Training, and Evaluation Configuration

[中文](training_zh.md) | [Configuration index](README.md) | [Back to README](../../README.md)

This guide covers the non-model portions of the composed Hydra configuration. Defaults come from `configs/train.yaml`; dataset and task recipes override them for a benchmark.

## Data configuration

`configs/data/libero_2cam.yaml` and `configs/data/robotwin.yaml` instantiate `RobotVideoDataset` and `WAMProcessor`. Keep these groups consistent when adding or modifying data:

| Group | Important settings |
| --- | --- |
| Dataset location | `dataset_dirs`, `text_embedding_cache_dir`, optional `pretrained_norm_stats` |
| Observation shape | `shape_meta.images`, `video_size`, `concat_multi_camera`, `num_output_cameras` |
| Sequence sampling | `num_frames`, `global_sample_stride`, `action_video_freq_ratio` |
| Robot tensors | `shape_meta.action`, `shape_meta.state`, `action_output_dim`, `proprio_output_dim` |
| Normalization | `norm_default_mode`, `norm_exception_mode`, `use_stepwise_action_norm`, optional transforms |
| Split behavior | `val_set_proportion`, `is_training_set`, `skip_padding_as_possible` |

`num_frames` is the action/state sequence length. `action_video_freq_ratio` sparsely selects video timestamps from that sequence; for example, the checked-in 33-step datasets with ratio `4` use 32 future actions and 9 video frames including the observation frame. Changing either value changes the training tensor shapes and must remain compatible with evaluation horizons.

Each image entry has a source `raw_shape` and a post-transform `shape`. The transform resize, `shape`, final `video_size`, camera concatenation mode, and `num_output_cameras` must describe the same layout. LIBERO concatenates two 224×224 cameras horizontally; RoboTwin uses its three-camera composition.

Action and proprio dimensions flow into the model through Hydra interpolation. If a processor changes dimensions or action representation, update `shape_meta` and the corresponding processor output dimension together.

### Normalization statistics

- LIBERO defaults to `min/max` normalization and can compute statistics from the configured training data.
- RoboTwin defaults to `z-score` and loads checked-in dataset statistics through `pretrained_norm_stats`.
- Evaluation must use the statistics associated with the trained checkpoint. Pass `EVALUATION.dataset_stats_path` when it cannot be discovered automatically.

### Text embeddings

Training normally leaves `model.backbone.load_text_encoder=false` and reads embeddings from `text_embedding_cache_dir`. Generate the cache with the same task/backbone selection:

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
```

Models sharing the same `text_encoder_id` and compatible context length can share a cache. Do not reuse embeddings across different encoders merely because the prompt text is identical.

## Training configuration

| Setting | Default | Behavior |
| --- | ---: | --- |
| `batch_size` | `16` | Per-process DataLoader batch size before gradient accumulation. Task recipes may override it. |
| `gradient_accumulation_steps` | `1` | Optimizer accumulation steps; effective global batch is per-process batch × process count × accumulation. |
| `learning_rate` | `1e-4` | AdamW learning rate. |
| `weight_decay` | `0.0` | AdamW weight decay; checked-in tasks commonly override it. |
| `lr_scheduler_type` | `cosine` | Trainer learning-rate schedule. |
| `warmup_ratio` | `0.05` | Fraction of steps used for warmup; must be in `[0, 1)`. |
| `max_steps` | `1000` | Required positive optimizer-step limit; epoch-based training is not used. |
| `mixed_precision` | `bf16` | One of `no`, `fp16`, or `bf16`. |
| `max_grad_norm` | `1.0` | Gradient clipping threshold. |
| `seed` | `42` | Training and worker seed. |
| `resume` | `null` | Resume input; leave null for a fresh run. |

The launcher controls distributed execution, while Hydra controls training behavior:

```bash
# Single node, 8 processes, ZeRO-1
NPROC_PER_NODE=8 bash scripts/train_zero1.sh task=libero_easywam_mot_wan22

# Two nodes, 8 processes per node, ZeRO-2
NNODES=2 NODE_RANK=0 MASTER_ADDR=<host> MASTER_PORT=29500 \
  NPROC_PER_NODE=8 bash scripts/train_zero2.sh task=robotwin_easywam_mot_wan22
```

Set the second node's `NODE_RANK=1`. All nodes must use the same `NNODES`, address, port, task, and overrides. Launch scripts set an output directory and WandB name from the selected task unless explicitly overridden later on the command line.

## Logging, checkpoints, and training-time evaluation

| Setting | Purpose |
| --- | --- |
| `output_dir` | Checkpoints, logs, and evaluation artifacts for the run. |
| `log_every` | Step interval for training metrics. |
| `save_every` | Step interval for checkpoint saving. |
| `eval_every` | Step interval for validation inference when a validation dataset exists. |
| `eval_num_inference_steps` | Denoising steps used by training-time evaluation. |
| `eval_save_video` | Save one stitched prediction/VAE/ground-truth video per rank at evaluation. |
| `wandb.*` | Enable WandB and set workspace, project, run name, group, and mode. |

Choose positive intervals relative to `max_steps`, or set an interval to `0` to disable that behavior. Video saving is useful for qualitative checks but adds decoding, synchronization, and storage cost.

## Evaluation configuration

Evaluation roots are `configs/sim_libero.yaml`, `configs/sim_libero_plus.yaml`, and `configs/sim_robotwin.yaml`. They inherit `train.yaml`, select a task, enable the runtime text encoder, skip redundant base-DiT initialization, and load the trained weights from `ckpt`.

Common policy controls include:

| Setting | Meaning |
| --- | --- |
| `EVALUATION.action_horizon` | Number of actions predicted; `null` derives it from the configured data sequence. |
| `EVALUATION.replan_steps` | Maximum actions executed before requesting a new prediction. |
| `EVALUATION.num_inference_steps` | Denoising steps for video/action inference. |
| `EVALUATION.sigma_shift` | Optional inference scheduler shift override. |
| `EVALUATION.text_cfg_scale` / `negative_prompt` | Text classifier-free guidance inputs. |
| `EVALUATION.dataset_stats_path` | Normalization statistics paired with the checkpoint. |
| `EVALUATION.device` / `rand_device` | Model execution and initial random-noise devices. |
| `EVALUATION.output_dir` | Result, log, summary, and rollout output root. |

`replan_steps` is bounded by the predicted action horizon at runtime. Shorter replanning reacts more frequently but invokes the model more often. `num_inference_steps` must be positive; reducing it trades denoising compute for possible policy-quality loss.

`MULTIRUN.num_gpus` and `MULTIRUN.max_tasks_per_gpu` control manager-side task sharding rather than training world size. See the [LIBERO](../instructions/libero.md), [LIBERO-Plus](../instructions/libero_plus.md), and [RoboTwin](../instructions/robotwin.md) guides for benchmark-specific selectors and resume rules.

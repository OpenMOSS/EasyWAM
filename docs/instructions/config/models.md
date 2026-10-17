# Model Configuration

[中文](models_zh.md) | [Configuration index](README.md) | [Back to README](../../../README.md)

Model configurations in `configs/model/` select an EasyWAM architecture, a visual backbone, and optional LoRA adapters. Task configurations combine a model with a dataset and training settings. Model names follow `easywam_<architecture>_<backbone>[_lora]`.

## Select an architecture

| Architecture | Behavior | Main settings |
| --- | --- | --- |
| Unified | A shared visual transformer jointly models visual and action tokens. | `action_dim`, `state_dim`, `projector_hidden_dim` |
| MoT | Visual and action experts interact through mixed self-attention. | `action_dit_config`, `action_dit_pretrained_path` |
| MoT-Joint | The two experts jointly denoise future observations and actions. | Same structural settings as MoT |
| MoT-IDM | Uses a teacher-forced future observation to condition action training. | `video_cond_noise_prob` |
| Hidden | Conditions ActionDiT on intermediate visual features. | `video_hidden_layer`, `detach_video_hidden`, `action_dit_config` |

Select an existing task, or override its model group while retaining the dataset and training settings:

```bash
python scripts/train.py --cfg job task=libero_easywam_mot_joint_cosmos25

python scripts/train.py --cfg job \
  task=libero_easywam_mot_wan22 \
  model=easywam_hidden_wan22_lora
```

## Select a backbone

| Backbone | Unified | MoT | MoT-Joint | MoT-IDM | Hidden |
| --- | :---: | :---: | :---: | :---: | :---: |
| Wan2.2-TI2V-5B | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cosmos-Predict2.5-2B | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX.2 Klein Base 4B | ✅ | ✅ | ✅ | ✅ | ✅ |

Backbone configurations define model paths, text and attention dimensions, scheduler defaults, and LoRA target modules. Model configurations select the appropriate transformer structure and derive its dimensions from the backbone. Weight downloads, text encoders, visual prediction targets, and backbone-specific constraints are described in the [backbone guides](../backbone/README.md).

Training normally uses cached text features with `model.backbone.load_text_encoder=false`. Text precomputation and benchmark evaluation enable the text encoder when runtime instructions need encoding.

## Data-derived dimensions

Robot dimensions come from the data processor:

```yaml
state_dim: ${data.train.processor.proprio_output_dim}
state_position: context
action_dit_config:
  action_dim: ${data.train.processor.action_output_dim}
```

Unified and Hidden also expose `action_dim` at the model level. When adding a dataset, keep `shape_meta`, `action_output_dim`, and `proprio_output_dim` consistent, then inspect the composed configuration with `--cfg job`.

## Configure ActionDiT

MoT-family and Hidden models use `model.action_dit_config` to configure their action expert. Unified projects actions into the shared transformer and has no separate ActionDiT.

The `preset` field selects a default block structure: `standard` uses a stack of sequential transformer blocks; `staged` uses sequential blocks followed by parallel attention/FFN blocks. Explicit component settings override the preset defaults.

| Option | Values | `standard` default | `staged` default |
| --- | --- | --- | --- |
| `cross_attention` | `true`, `false` | true | false |
| `qkv_projection` | `separate`, `fused` | separate | fused |
| `qkv_bias` | `true`, `false` | true | false |
| `qk_norm` | `projection`, `head`, `none` | projection | head |
| `ffn_activation` | `gelu`, `swiglu` | gelu | swiglu |
| `encoder_type` | `mlp`, `linear` | mlp | linear |
| `decoder_type` | `mlp`, `linear`, `time_modulated` | mlp | time_modulated |
| `position_encoding` | `rope_1d`, `rope_nd` | rope_1d | rope_nd |
| `time_embedding_type` | `standard`, `scaled` | standard | scaled |

`num_layers` controls standard depth; `num_layers_double` and `num_layers_single` control the two staged depths. `hidden_dim` sets the action width, `ffn_dim` or `mlp_ratio` sets the FFN width, and `max_action_horizon` sets the maximum action sequence length. In staged parallel blocks, the fused QKV and FFN input share a projection, so `qkv_bias` also controls that FFN input bias. Standard blocks use one-dimensional RoPE.

MoT-family experts need compatible block topology and attention dimensions. Hidden reads visual features through cross-attention and requires `cross_attention=true`. Use the selected model recipe as the starting point, and use initialization weights that match any structural changes.

## Initialize and load weights

| Setting | Meaning |
| --- | --- |
| Backbone path fields | Pretrained visual transformer, VAE, and text-encoder locations. |
| `action_dit_pretrained_path` | Optional initialization for a separate ActionDiT; `null` uses random initialization. |
| `skip_dit_load_from_pretrain` | Skip visual pretrained weights before loading an EasyWAM checkpoint. |
| `ckpt` | Policy checkpoint used by benchmark evaluation. |
| `resume` | Training checkpoint or directory used to restore training state. |

`scripts/preprocess_action_dit_backbone.py` prepares ActionDiT initialization from visual transformer weights. The script reads the backbone selected by `--model-config`; optional `--backbone` checks that selection. Follow the selected backbone guide for the model configuration and output path. Unified does not require this step. When loading a policy checkpoint, use the matching architecture, state layout, and action structure.

## Full training and LoRA

Select a full model recipe for full training, or its `_lora` counterpart for visual-backbone LoRA. LoRA configurations compose `configs/model/lora/video_dit.yaml`; the default rank and alpha are `128`, with zero dropout and backbone-specific target modules.

```bash
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_mot_wan22 \
  model=easywam_mot_wan22_lora \
  model.lora.r=64 \
  model.lora.lora_alpha=64
```

Use the same task and model selection for text precomputation, training, and evaluation. The [training guide](training.md) covers optimizer settings, checkpointing, and resume.

## Loss weights and schedulers

The total objective is `lambda_video * loss_video + lambda_action * loss_action`. Both weights are configured under `model.loss`:

```yaml
loss:
  lambda_video: 1.0
  lambda_action: 1.0
```

Defaults come from the selected model recipe. Set a weight to `0` to disable that objective.

`model.backbone.video_scheduler.family` selects `continuous_flow` or `flow_unipc`, with defaults supplied by the backbone. `video_scheduler` comes from the backbone; models with a separate action expert also define `action_scheduler`. `train_shift` controls training timestep sampling, `infer_shift` controls inference, and `num_train_timesteps` defines the timestep range.

## Conditioning and attention

| Setting | Usage |
| --- | --- |
| `state_position` | `context` adds state to the conditioning context; `sequence` places state after action tokens in the modeled sequence. Use the same layout when loading a checkpoint. |
| `video_cond_noise_prob` | Probability of adding noise to the teacher-forced observation in MoT-IDM; range `[0, 1]`, default `0.5`. |
| `video_hidden_layer` | Zero-based visual block index read by Hidden; defaults to the backbone setting. |
| `detach_video_hidden` | `true` stops action-loss gradients through Hidden's visual features; `false` permits them. |
| `projector_hidden_dim` | Hidden width of the state/action projection MLPs. |
| `video_attention_mask_mode` | Visual attention layout where supported by the backbone; see its guide for available modes. |

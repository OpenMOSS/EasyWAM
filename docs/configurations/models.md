# Model Configuration

[中文](models_zh.md) | [Configuration index](README.md) | [Back to README](../../README.md)

Model recipes live in `configs/model/`. Their names follow `easywam_<architecture>_<backbone>[_lora]`. Task recipes select one of these models and provide the data dimensions referenced by the model config.

## Select an architecture

| Architecture | Configuration-specific behavior | Architecture-specific settings |
| --- | --- | --- |
| EasyWAM-Unified | One Video DiT jointly represents video, action, and state tokens. | `action_dim`, `state_dim`, `projector_hidden_dim` |
| EasyWAM-MoT | Separate Video DiT and Action DiT experts interact through mixed self-attention. | `action_dit_config`, `action_dit_pretrained_path` |
| EasyWAM-MoT-Joint | The two experts jointly denoise future video and action tokens. | Same structural settings as MoT |
| EasyWAM-MoT-IDM | Adds a teacher-forced conditional-video branch for action prediction. | `video_cond_noise_prob` in `[0, 1]` |
| EasyWAM-Hidden | Conditions Action DiT on an intermediate Video DiT hidden layer. | `video_hidden_layer`, `detach_video_hidden` |

Use the corresponding task rather than editing `_target_` manually:

```bash
# MoT-Joint + Cosmos2.5 on LIBERO
python scripts/train.py --cfg job task=libero_easywam_mot_joint_cosmos25

# Hidden + Wan2.2 with LoRA on RoboTwin
python scripts/train.py --cfg job task=robotwin_easywam_hidden_wan22_lora
```

## Backbone support

| Backbone | Unified | MoT | MoT-Joint | MoT-IDM | Hidden |
| --- | :---: | :---: | :---: | :---: | :---: |
| Wan2.2-TI2V-5B | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cosmos-Predict2.5-2B | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX.2 Klein-4B | — | ✅ | — | — | — |

Backbone configs contain model/checkpoint paths, text dimensions, transformer dimensions, scheduler defaults, attention settings, and the list of LoRA target modules. Keep the action expert's layer count, number of heads, and head dimension aligned with the video expert; checked-in recipes derive these values from `model.backbone`.

`model.backbone.load_text_encoder` is `false` for training because datasets load precomputed embeddings. Evaluation configs set it to `true` so policies can encode runtime task instructions.

See the dedicated [FLUX.2 / ImageWAM integration guide](../instructions/flux2_imagewam_integration.md) for its source checkout, checkpoint migration, and Qwen3 cache requirements.

## Data-derived dimensions

Do not normally hard-code robot dimensions in a model recipe:

```yaml
state_dim: ${data.train.processor.proprio_output_dim}
action_dit_config:
  action_dim: ${data.train.processor.action_output_dim}
```

Unified and Hidden expose `action_dim` at the model level; MoT-family models place it under `action_dit_config`. When adding a dataset, update `shape_meta`, `action_output_dim`, and `proprio_output_dim` together, then inspect the composed config.

## Checkpoint initialization

| Setting | Meaning |
| --- | --- |
| Backbone path fields | Locate the pretrained Video DiT, VAE/tokenizer, and optional text encoder. |
| `action_dit_pretrained_path` | Optional interpolated ActionDiT initialization used by the MoT family and Hidden recipes. |
| `skip_dit_load_from_pretrain` | Skip loading pretrained DiT weights; evaluation sets this to `true` before loading the EasyWAM checkpoint. |
| `ckpt` | Evaluation checkpoint path, defined by `configs/sim_*.yaml`. |
| `resume` | Training checkpoint/directory consumed by the trainer's resume workflow. |

Use `scripts/preprocess_action_dit_backbone.py` to build the Wan2.2 or Cosmos2.5 ActionDiT initialization described in the root README. EasyWAM-Unified does not use a separate ActionDiT checkpoint.

## Full training and LoRA

A `_lora` model recipe adds `configs/model/lora/video_dit.yaml`. The default adapter uses rank and alpha `128`, zero dropout, and backbone-specific target modules. Choose LoRA by selecting a `_lora` task; do not attach the group to an already trained full-model checkpoint unless that checkpoint's loading contract expects adapters.

Override LoRA parameters when composing the run:

```bash
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_mot_wan22_lora \
  model.lora.r=64 \
  model.lora.lora_alpha=64
```

## Loss weights and schedulers

All architecture configs expose:

```yaml
loss:
  lambda_video: 1.0
  lambda_action: 1.0
```

The total objective is `lambda_video * loss_video + lambda_action * loss_action`. Set a weight to `0` only when intentionally disabling that objective; keep both explicit in new recipes. FLUX.2 MoT currently defaults `lambda_video` to `0.5`.

`video_scheduler` is inherited from the backbone. Architectures with a separate ActionDiT also define `action_scheduler`. `train_shift` affects training noise/timestep sampling, `infer_shift` affects inference, and `num_train_timesteps` defines the scheduler discretization. Treat scheduler changes as model-behavior changes rather than generic speed tuning.

## Architecture-specific settings

- `video_cond_noise_prob` controls how often MoT-IDM adds noise to the teacher-forced conditional video during training. It must be between `0` and `1` and defaults to `0.5`.
- `video_hidden_layer` is a zero-based Video DiT block index used by Hidden. The checked-in value comes from the backbone and must remain within its layer count.
- `detach_video_hidden=true` prevents the action loss from backpropagating through Hidden's video feature path. Setting it to `false` couples action gradients into that path and changes training memory and optimization behavior.
- `projector_hidden_dim` controls the small state/action projection MLPs; changing it makes checkpoint shapes incompatible.
- `video_attention_mask_mode` is a backbone behavior setting. Checked-in Wan2.2 and Cosmos2.5 recipes use `first_frame_causal`; do not change it when resuming a checkpoint trained with another mode.

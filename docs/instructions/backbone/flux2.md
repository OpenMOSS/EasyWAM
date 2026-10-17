# FLUX.2 Klein Base 4B Backbone

[中文](flux2_zh.md) | [Backbone index](README.md) | [Model configuration](../config/models.md) | [Back to project README](../../../README.md)

FLUX.2 supports Unified, MoT, MoT-Joint, MoT-IDM, and Hidden, with full-parameter and LoRA model configurations. The backbone combines a double-stream/single-stream DiT, an image VAE, and a Qwen3 text encoder. Default paths and runtime settings are in `configs/model/backbone/flux2_klein_4b.yaml`.

## Prepare model weights

Run from the project root. Accept the autoencoder repository's access conditions on Hugging Face and run `huggingface-cli login` before downloading gated files:

```bash
mkdir -p checkpoints/flux2

huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B \
  --include "flux-2-klein-base-4b.safetensors" \
  --local-dir checkpoints/flux2/FLUX.2-klein-base-4B

huggingface-cli download black-forest-labs/FLUX.2-dev \
  --include "ae.safetensors" \
  --local-dir checkpoints/flux2/FLUX.2-dev

huggingface-cli download Qwen/Qwen3-4B \
  --local-dir checkpoints/Qwen3-4B
```

The text encoder defaults to `Qwen/Qwen3-4B`. To use the local download, pass `model.backbone.qwen3_model_spec=./checkpoints/Qwen3-4B` when precomputing text or evaluating. Transformer and VAE locations are configured with `model.backbone.model_path` and `model.backbone.ae_model_path`.

## Initialize ActionDiT

Action experts use the `staged` preset, width `1024`, five sequential blocks, and twenty parallel blocks. Hidden enables cross-attention to visual features and also supports an independently configured action depth. The general component options are documented in [model configuration](../config/models.md#configure-actiondit).

`model.action_dit_pretrained_path=null` selects random initialization. To initialize a staged ActionDiT from visual weights:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --backbone flux2 \
  --model-config configs/model/easywam_mot_flux2_klein_4b.yaml \
  --output checkpoints/ActionDiT_FLUX2_4B_alphascale_1024hdim.pt
```

Pass `model.action_dit_pretrained_path=./checkpoints/ActionDiT_FLUX2_4B_alphascale_1024hdim.pt` to training. For Hidden, use `configs/model/easywam_hidden_flux2_klein_4b.yaml` and a separate output file. MoT-family default action structures share an initialization file; Unified has no separate ActionDiT.

Alpha scaling is enabled by default (`--apply-alpha-scaling true`): weights resized along the input dimension are scaled by `sqrt(source_dim / target_dim)`. The converter transfers compatible visual parameters and resizes Q/K/V and gated-FFN sections separately. Its output includes parameter sources and coverage; new conditioning modules retain their initialization. Conversion supports the staged preset, and the resulting file must match the selected action structure.

## Prepare text embeddings

After [preparing the dataset](../data/README.md), generate its Qwen3 feature cache:

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_flux2_klein_4b
```

For another dataset, select its task and override the model group, for example:

```bash
python scripts/precompute_text_embeds.py \
  task=vlabench_easywam_mot_wan22 \
  model=easywam_mot_flux2_klein_4b
```

The task determines the dataset and cache destination. Features are shared across the five architectures when the dataset, encoder settings, and `context_len` match. Cache files follow the shared `<sha256>.text_len<context_len>.<encoder_id>[dtype].pt` naming and contain `context`, a boolean `mask`, and cache metadata. Keep cache paths and text settings consistent between precomputation and training. Caches using the earlier dedicated format must be regenerated with this command.

## Train

Select the dataset task and model configuration:

```bash
# Full-parameter MoT training
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_flux2_klein_4b

# Hidden with visual-backbone LoRA
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_mot_flux2_klein_4b \
  model=easywam_hidden_flux2_klein_4b_lora
```

Model names follow `easywam_<architecture>_flux2_klein_4b`, with `unified`, `mot`, `mot_joint`, `mot_idm`, or `hidden` as the architecture; append `_lora` for LoRA. Model recipes default to `lambda_video=0.5` and `lambda_action=1.0`.

### Prediction and conditioning

Training uses the shared `video`, `action`, `proprio`, `context`, and `context_mask` fields. The first and last video frames provide the current observation and future endpoint. Visual loss and evaluation use the future endpoint; returned visualizations contain the current image and predicted endpoint.

| Architecture | Action conditioning and inference |
| --- | --- |
| MoT | Actions read text, the current image, and state. Action inference can cache the causal visual prefix. |
| MoT-Joint | Jointly denoises the endpoint and actions. |
| MoT-IDM | Uses a separate teacher endpoint during training; inference generates the endpoint before actions. |
| Hidden | Reads intermediate visual features. Action inference extracts them once from the current image and future noise; joint inference also denoises the endpoint while reusing the initial conditioning features. |
| Unified | Jointly denoises endpoint and action/state tokens through the shared transformer. |

`video_hidden_layer` indexes blocks across both visual stages. Hidden requires cross-attention but can use either action preset; visual-to-action weight conversion requires staged blocks. Both `context` and `sequence` state layouts are supported. Use `text_cfg_scale=1` and no negative prompt. Policy loading checks the saved architecture and action structure; legacy ImageWAM policy checkpoints are not supported.

## Evaluate

Use the same task and model selection as training, together with the matching dataset statistics:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_flux2_klein_4b \
  model=easywam_hidden_flux2_klein_4b_lora \
  ckpt=<path/to/checkpoint.pt> \
  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json>
```

Evaluation loads the Qwen3 text encoder for runtime instructions. Simulator setup, rollout options, and result layout are covered by the [benchmark guides](../benchmark/README.md).

## Standalone visual training

`Flux2Core` trains the visual DiT without an action expert. Prepare the weights, dataset, and text cache above before running this single-step example from the project root. It uses the existing dataset protocol; action and state fields are optional and ignored. The VAE and text encoder remain frozen in evaluation mode. Cached text takes priority over prompts; enable `backbone.load_text_encoder` in the Python configuration to encode prompts when no cache is supplied.

```bash
PYTHONPATH=src:. python - <<'PYTHON'
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from model.backbone.flux2 import Flux2Core
from runtime import build_datasets
from utils import misc
from utils.config_resolvers import register_default_resolvers

register_default_resolvers()
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=libero_easywam_mot_flux2_klein_4b"])
misc.register_work_dir("./runs/flux2_core")
dataset, _ = build_datasets(cfg.data)
sample = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))

model = Flux2Core.from_flux2_pretrained(
    cfg.model.backbone, device="cuda", torch_dtype=torch.bfloat16,
)
optimizer = torch.optim.AdamW(model.dit.parameters(), lr=1e-4)
model.train()
optimizer.zero_grad(set_to_none=True)
loss, metrics = model(sample)
loss.backward()
optimizer.step()
model.save_checkpoint("./runs/flux2_core/checkpoint.pt", optimizer=optimizer, step=1)
model.load_checkpoint("./runs/flux2_core/checkpoint.pt", optimizer=optimizer)

result = model.infer(
    sample["video"][0, :, 0],
    context=sample["context"][:1],
    context_mask=sample["context_mask"][:1],
    num_frames=2, num_inference_steps=20, seed=42,
)
result["video"][-1].save("./runs/flux2_core/endpoint.png")
PYTHON
```

Training consumes `video` shaped `[B,3,T,H,W]`, with values in `[-1,1]`, `T >= 2`, and image dimensions divisible by 16. It uses the first and last frames, `context`/`context_mask` (or `prompt`), and optional `[B,T]` `image_is_pad`; padded target frames contribute zero loss. Inference accepts one image and produces the current image and future endpoint (`num_frames=2`), with `guidance_scale=1` and no negative conditions.

Core checkpoints store DiT weights, backbone/variant metadata, the step, and optional optimizer state. Recreate the Core with the same pretrained VAE and text configuration before restoring; these are visual checkpoints, while simulator evaluation uses EasyWAM policy checkpoints.

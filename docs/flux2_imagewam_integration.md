# FLUX.2 / ImageWAM backbone integration

This integration treats ImageWAM as an EasyWAM backbone recipe rather than a
second runtime. The shared loader returns `BackboneComponents`, and MoT selects
one explicit block protocol:

- `main`: the unchanged homogeneous block loop used by Wan2.2 and Cosmos2.5.
- `flux2`: FLUX.2 double-stream blocks followed by single-stream blocks.

The FLUX.2 component split is Qwen3 text encoder, official FLUX.2 autoencoder,
official Klein transformer video/image expert, and `ActionDiTFlux2`. Cached
ImageWAM Qwen3 tensors remain supported with the
`<sha256>.qwen3_flux2_len512.pt` filename and
`text_hidden_states`/`text_attention_mask` payload fields.

## Configure

Edit `configs/model/backbone/flux2_klein_4b.yaml` or override these values:

```text
model.backbone.flux2_src_path=/path/to/official/flux2/source
model.backbone.model_path=/path/to/flux-2-klein-base-4b.safetensors
model.backbone.ae_model_path=/path/to/ae.safetensors
data.train.text_embedding_cache_dir=/path/to/qwen3/cache
data.val.text_embedding_cache_dir=/path/to/qwen3/cache
```

Then launch, for example:

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=robotwin_easywam_mot_flux2_klein_4b \
  model.backbone.flux2_src_path=/path/to/flux2 \
  model.backbone.model_path=/path/to/flux2-klein-base-4b.safetensors \
  model.backbone.ae_model_path=/path/to/ae.safetensors \
  data.train.text_embedding_cache_dir=/path/to/qwen3/cache \
  data.val.text_embedding_cache_dir=/path/to/qwen3/cache
```

The training-side integration gate covers FLUX.2 image/action loss and exact
ImageWAM checkpoint migration. `EasyWAMMoT.infer_action()` also implements the
released policy's action-only closed-loop path: it encodes the Qwen3 context,
current image and proprioception once, caches the FLUX.2 prefix K/V tensors,
and denoises the action stream for the requested horizon.

The LIBERO and LIBERO-Plus evaluators can therefore use a released ImageWAM
checkpoint through the normal EasyWAM policy interface. The release contract
uses a 16-step action chunk, 12 executed steps before replanning, 10 denoising
steps, and the checkpoint's sibling `dataset_stats.json`. Always require exact
checkpoint coverage and retain the evaluator's per-task results and rollout
videos. A seven-category LIBERO-Plus `libero_spatial` integration subset passed
7/7 in the initial validation; this is a pipeline/compatibility gate, not a
replacement for the full benchmark.

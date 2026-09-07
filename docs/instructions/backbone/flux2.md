# FLUX.2 Klein Base 4B / ImageWAM Backbone

[Backbone index](README.md) | [Backbone configuration](../config/models.md) | [Back to project README](../../../README.md)

EasyWAM integrates the official FLUX.2 double-stream and single-stream blocks with the ImageWAM-compatible ActionDiT and checkpoint contract. The checked-in recipes support EasyWAM-MoT on LIBERO and RoboTwin. The default paths and runtime settings live in `configs/model/backbone/flux2_klein_4b.yaml`.

## Prepare the backbone

Clone the official source tree at the path used by the default config:

```bash
git clone https://github.com/black-forest-labs/flux2.git third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15
```

Download the trainable Klein Base 4B checkpoint and the FLUX.2 autoencoder. The FLUX.2-dev repository is gated, so accept its license on Hugging Face and run `huggingface-cli login` first.

```bash
mkdir -p checkpoints/flux2

huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B \
  --include "flux-2-klein-base-4b.safetensors" \
  --local-dir checkpoints/flux2/FLUX.2-klein-base-4B

huggingface-cli download black-forest-labs/FLUX.2-dev \
  --include "ae.safetensors" \
  --local-dir checkpoints/flux2/FLUX.2-dev
```

The text encoder defaults to `Qwen/Qwen3-4B` and is downloaded by Transformers when evaluation first loads it. For an offline installation, download it explicitly and override the config:

```bash
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir checkpoints/Qwen3-4B
```

Then pass `model.backbone.qwen3_model_spec=./checkpoints/Qwen3-4B` to training and evaluation commands. Other non-default locations can be supplied with `model.backbone.flux2_src_path`, `model.backbone.model_path`, and `model.backbone.ae_model_path`.

## Prepare text embeddings

FLUX.2 training consumes the ImageWAM Qwen3 cache format. Clone [ImageWAM](https://github.com/yuyangalin/ImageWAM), then run its cache script against the prepared [LIBERO](../data/libero.md) or [RoboTwin](../data/robotwin.md) dataset. For example, from the ImageWAM root:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/flux2/precompute_flux2_qwen3_embeds.py \
  task=libero_flux2_imagewam \
  'data.train.dataset_dirs=[/absolute/path/to/libero_spatial,/absolute/path/to/libero_object,/absolute/path/to/libero_goal,/absolute/path/to/libero_10]' \
  data.train.qwen_text_cache_dir=/absolute/path/to/qwen3/cache \
  data.train.qwen_context_len=128 \
  data.train.qwen_text_cache_format=qwen3_flux2 \
  model.flux2_src_path=/absolute/path/to/third_party/flux2 \
  model.variant=klein-base-4b
```

Use `qwen_context_len=128` for the checked-in LIBERO recipe and `qwen_context_len=512` for the checked-in RoboTwin recipe. Point both EasyWAM dataset splits to the generated directory:

```text
data.train.text_embedding_cache_dir=/path/to/qwen3/cache
data.val.text_embedding_cache_dir=/path/to/qwen3/cache
```

Each cache file must be named `<sha256>.qwen3_flux2_len<context_len>.pt` and contain `text_hidden_states` with shape `[context_len, D]` plus a boolean `text_attention_mask` with shape `[context_len]`. `scripts/precompute_text_embeds.py` currently handles Wan2.2 and Cosmos2.5 only; it does not produce this format.

## Train

Pass the cache location required by the FLUX.2 task recipe:

```bash
# LIBERO
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_flux2_klein_4b \
  data.train.text_embedding_cache_dir=/path/to/qwen3/cache \
  data.val.text_embedding_cache_dir=/path/to/qwen3/cache

# RoboTwin
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=robotwin_easywam_mot_flux2_klein_4b \
  data.train.text_embedding_cache_dir=/path/to/qwen3/cache \
  data.val.text_embedding_cache_dir=/path/to/qwen3/cache
```

FLUX.2 currently trains one endpoint image. Its MoT implementation uses Qwen3 text features, the FLUX.2 autoencoder, the official Klein image expert, and `ActionDiTFlux2`; it does not require `scripts/preprocess_action_dit_backbone.py`.

## Evaluate

Use the matching FLUX.2 task recipe and an EasyWAM or ImageWAM-compatible checkpoint:

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_flux2_klein_4b \
  ckpt=<path/to/checkpoint.pt>

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_flux2_klein_4b \
  ckpt=<path/to/checkpoint.pt>
```

The action-only closed-loop path encodes text, the current image, and proprioception once, caches the FLUX.2 prefix K/V tensors, and denoises the requested action horizon. ImageWAM checkpoints are migrated during loading and must have exact tensor coverage. Use the checkpoint's matching `dataset_stats.json`.

Follow the [LIBERO evaluation guide](../benchmark/libero.md), [LIBERO-Plus guide](../benchmark/libero_plus.md), or [RoboTwin evaluation guide](../benchmark/robotwin.md) for simulator setup, batching, and result layout. The ImageWAM release contract uses a 16-step action chunk, executes 12 steps before replanning, and uses 10 denoising steps; pass those values explicitly when reproducing that policy.

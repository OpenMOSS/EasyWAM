# FLUX.2 Klein Base 4B / ImageWAM Backbone

[Backbone index](README.md) | [Backbone configuration](../config/models.md) | [Back to project README](../../../README.md)

EasyWAM integrates the official FLUX.2 double-stream and single-stream blocks with the ImageWAM-compatible ActionDiT and checkpoint contract. The checked-in training recipes support EasyWAM-MoT on LIBERO, RoboTwin, RoboCasa365, and RoboDojo. The default paths and runtime settings live in `configs/model/backbone/flux2_klein_4b.yaml`.

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

The text encoder defaults to `Qwen/Qwen3-4B` and is downloaded by Transformers when text precomputation or evaluation first loads it. For an offline installation, download it explicitly and override the config:

```bash
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir checkpoints/Qwen3-4B
```

Then pass `model.backbone.qwen3_model_spec=./checkpoints/Qwen3-4B` to training and evaluation commands. Other non-default locations can be supplied with `model.backbone.flux2_src_path`, `model.backbone.model_path`, and `model.backbone.ae_model_path`.

## Prepare text embeddings

FLUX.2 training consumes the ImageWAM-compatible Qwen3 cache format. Use the same EasyWAM precomputation command as for Wan2.2 and Cosmos2.5, selecting the training task for your dataset:

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_flux2_klein_4b
```

The precomputation script is dataset-agnostic; create a matching FLUX.2 task recipe as described in the [configuration guide](../config/README.md) when using another dataset. Each task inherits its cache directory from its data configuration, and train/validation splits share it where applicable. LIBERO uses 128 tokens; RoboTwin, RoboCasa365, and RoboDojo use 512. A custom cache location can be supplied with `data.train.text_embedding_cache_dir` and, for datasets with a validation split, `data.val.text_embedding_cache_dir` in both precomputation and training commands.

Each cache file is named `<sha256>.qwen3_flux2_len<context_len>.pt` and contains `text_hidden_states` with shape `[context_len, D]` plus a boolean `text_attention_mask` with shape `[context_len]`.

## Train

After precomputation, train using the matching task. For example:

```bash
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_flux2_klein_4b
```

FLUX.2 currently trains one endpoint image. Its MoT implementation uses Qwen3 text features, the FLUX.2 autoencoder, the official Klein image expert, and `ActionDiTFlux2`; it does not require `scripts/preprocess_action_dit_backbone.py`.

## Evaluate

Use the matching FLUX.2 task recipe and an EasyWAM or ImageWAM-compatible checkpoint:

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_flux2_klein_4b \
  ckpt=<path/to/checkpoint.pt>
```

The action-only closed-loop path encodes text, the current image, and proprioception once, caches the FLUX.2 prefix K/V tensors, and denoises the requested action horizon. ImageWAM checkpoints are migrated during loading and must have exact tensor coverage. Use the checkpoint's matching `dataset_stats.json`.

Follow the [LIBERO evaluation guide](../benchmark/libero.md) or [LIBERO-Plus guide](../benchmark/libero_plus.md) for simulator setup, batching, and result layout. The ImageWAM release contract uses a 16-step action chunk, executes 12 steps before replanning, and uses 10 denoising steps; pass those values explicitly when reproducing that policy.

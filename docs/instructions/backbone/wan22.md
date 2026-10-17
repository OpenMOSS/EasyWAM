# Wan2.2-TI2V-5B Backbone

[中文](wan22_zh.md) | [Backbone index](README.md) | [Model configuration](../config/models.md) | [Back to project README](../../../README.md)

Wan2.2 is supported by EasyWAM-Unified, MoT, MoT-Joint, MoT-IDM, and Hidden, with full-parameter and LoRA model configurations. The default paths and runtime settings live in `configs/model/backbone/wan22.yaml`.

## Prepare model weights

Run from the project root after installing EasyWAM and the Hugging Face CLI:

```bash
mkdir -p checkpoints
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B
```

This directory supplies the Video DiT, VAE, UMT5 encoder, and tokenizer. The default config expects the UMT5 files under `checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl`.

## Initialize ActionDiT

MoT, MoT-Joint, MoT-IDM, and Hidden use a separate ActionDiT. Generate its interpolated initialization once:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot_wan22.yaml \
  --backbone wan22 \
  --output checkpoints/ActionDiT_Wan22_5B_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified does not use this ActionDiT file. To store weights elsewhere, override `model.backbone.model_id`, `model.backbone.tokenizer_model_id`, or `model.backbone.action_dit_pretrained_path` instead of editing shared configs.

## Prepare text embeddings

After preparing the [LIBERO](../data/libero.md) or [RoboTwin](../data/robotwin.md) dataset, generate the UMT5 cache with a matching task recipe:

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_wan22
```

The selected task determines the dataset directories and cache destination. The same Wan2.2 cache is reusable across EasyWAM architectures when the dataset and `context_len` are unchanged. Use `torchrun --standalone --nproc_per_node=<gpu-count>` before the script for multi-GPU preprocessing.

## Train

Select a checked-in task named `<benchmark>_easywam_<architecture>_wan22`. LIBERO also provides `_lora` task recipes. For example:

```bash
# Full-parameter MoT training on LIBERO
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_wan22

# LoRA Unified training on LIBERO
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_unified_wan22_lora
```

Available architecture segments are `unified`, `mot`, `mot_joint`, `mot_idm`, and `hidden`. Training outputs are written to `output_dir`, which defaults to `runs/train/<date>_<time>/`.

The model uses the `standard` ActionDiT preset. Visual attention is controlled by `model.backbone.video_attention_mask_mode`: `first_frame_causal` (default), `per_frame_causal`, or `bidirectional`. In `per_frame_causal` mode, tokens within a frame interact bidirectionally and each frame sees itself and earlier frames.

## Evaluate

Use the same task recipe as the checkpoint so model dimensions and backbone settings match:

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt>

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_easywam_mot_wan22 \
  ckpt=<path/to/checkpoint.pt>
```

Evaluation loads the runtime text encoder automatically. Follow the [LIBERO evaluation guide](../benchmark/libero.md), [LIBERO-Plus guide](../benchmark/libero_plus.md), or [RoboTwin evaluation guide](../benchmark/robotwin.md) for simulator setup, normalization statistics, batching, and result layout.

## Standalone visual training

`Wan22Core` is exported by `model.backbone.wan22`. Load weights with `Wan22Core.from_wan22_pretrained(...)`, call `model(sample)` for visual loss, and use `infer`, `save_checkpoint`, and `load_checkpoint` for generation and restoration. Samples contain normalized `video` (`[B,3,T,H,W]`) and `prompt`; `T % 4 == 1` and both spatial dimensions must be multiples of 16. This Core encodes prompts at runtime and does not consume the policy text cache. Action and state fields are not used by the Core. VAE and text encoder parameters should be frozen when optimizing only `model.dit`.

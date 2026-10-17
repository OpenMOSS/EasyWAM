# Cosmos-Predict2.5-2B Backbone

[中文](cosmos25_zh.md) | [Backbone index](README.md) | [Model configuration](../config/models.md) | [Back to project README](../../../README.md)

Cosmos2.5 is supported by EasyWAM-Unified, MoT, MoT-Joint, MoT-IDM, and Hidden, with full-parameter and LoRA model configurations. The default paths and runtime settings live in `configs/model/backbone/cosmos25.yaml`.

## Prepare model weights

The video model supplies the post-trained 2B DiT and its Wan2.1 tokenizer. Cosmos-Reason1-7B supplies the text encoder and tokenizer used for offline caching and runtime instructions:

```bash
mkdir -p checkpoints

huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --include "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt" "tokenizer.pth" \
  --local-dir checkpoints/Cosmos-Predict2.5-2B

huggingface-cli download nvidia/Cosmos-Reason1-7B \
  --local-dir checkpoints/Cosmos-Reason1-7B
```

## Initialize ActionDiT

MoT, MoT-Joint, MoT-IDM, and Hidden use a separate ActionDiT. Generate its interpolated initialization once:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot_cosmos25.yaml \
  --backbone cosmos25 \
  --output checkpoints/ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified does not use this ActionDiT file. To store weights elsewhere, override `model.backbone.model_id`, `model.backbone.reason_model_id`, or `model.backbone.action_dit_pretrained_path`.

## Prepare text embeddings

After preparing the [LIBERO](../data/libero.md) dataset, generate the projected Cosmos-Reason cache with a matching task recipe:

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_cosmos25
```

The selected task determines the dataset directories and cache destination. The same Cosmos2.5 cache is reusable across EasyWAM architectures when the dataset and `context_len` are unchanged. Cosmos encoding uses substantial memory and processes one prompt per device; use `torchrun --standalone --nproc_per_node=<gpu-count>` before the script to distribute prompts across GPUs.

## Train

Select a checked-in LIBERO task named `libero_easywam_<architecture>_cosmos25`; append `_lora` for LoRA. For example:

```bash
# Full-parameter MoT-Joint training on LIBERO
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_joint_cosmos25

# LoRA Hidden training on LIBERO
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_hidden_cosmos25_lora
```

Available architecture segments are `unified`, `mot`, `mot_joint`, `mot_idm`, and `hidden`. Training outputs are written to `output_dir`, which defaults to `runs/train/<date>_<time>/`.

The model uses the `standard` ActionDiT preset. Visual attention is controlled by `model.backbone.video_attention_mask_mode`: `first_frame_causal` (default), `per_frame_causal`, or `bidirectional`. In `per_frame_causal` mode, tokens within a frame interact bidirectionally and each frame sees itself and earlier frames.

## Evaluate

Use the same task recipe as the checkpoint so model dimensions and backbone settings match:

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_cosmos25 \
  ckpt=<path/to/checkpoint.pt>
```

Evaluation loads Cosmos-Reason1-7B automatically. Follow the [LIBERO evaluation guide](../benchmark/libero.md) or [LIBERO-Plus guide](../benchmark/libero_plus.md) for simulator setup, normalization statistics, batching, and result layout.

## Standalone visual training

`Cosmos25Core` is exported by `model.backbone.cosmos25`. Load weights with `Cosmos25Core.from_cosmos25_pretrained(...)`, call `model(sample)` for visual loss, and use `infer`, `save_checkpoint`, and `load_checkpoint` for generation and restoration. Samples contain normalized `video` (`[B,3,T,H,W]`) and either `prompt` or cached `context`/`context_mask`; omit `prompt` when passing cached features. Frame counts and image dimensions must satisfy the VAE constraints described above. Action and state fields are not required. VAE and text encoder parameters should be frozen when optimizing only `model.dit`.

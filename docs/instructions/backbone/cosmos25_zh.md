# Cosmos-Predict2.5-2B Backbone

[English](cosmos25.md) | [Backbone 索引](README_zh.md) | [模型配置](../config/models_zh.md) | [返回项目 README](../../../README_zh.md)

Cosmos2.5 支持 EasyWAM-Unified、MoT、MoT-Joint、MoT-IDM 和 Hidden，并提供全参数与 LoRA 模型配置。默认路径和运行设置位于 `configs/model/backbone/cosmos25.yaml`。

## 准备模型权重

视频模型提供 post-trained 2B DiT 及其 Wan2.1 tokenizer。Cosmos-Reason1-7B 提供离线文本缓存和运行时任务指令所需的文本编码器及 tokenizer：

```bash
mkdir -p checkpoints

huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --include "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt" "tokenizer.pth" \
  --local-dir checkpoints/Cosmos-Predict2.5-2B

huggingface-cli download nvidia/Cosmos-Reason1-7B \
  --local-dir checkpoints/Cosmos-Reason1-7B
```

## 初始化 ActionDiT

MoT、MoT-Joint、MoT-IDM 和 Hidden 使用独立的 ActionDiT。生成一次插值初始化权重：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot_cosmos25.yaml \
  --backbone cosmos25 \
  --output checkpoints/ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified 不使用该文件。自定义权重位置时，可覆盖 `model.backbone.model_id`、`model.backbone.reason_model_id` 或 `model.backbone.action_dit_pretrained_path`。

## 预计算文本特征

准备好 [LIBERO](../data/libero_zh.md) 数据集后，选择对应 task 生成投影后的 Cosmos-Reason 文本缓存：

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_cosmos25
```

Task 决定数据集目录和缓存位置。数据集与 `context_len` 相同时，不同 EasyWAM 架构可复用同一份 Cosmos2.5 文本缓存。文本编码显存占用较高，每个设备逐条处理 prompt；可在脚本前使用 `torchrun --standalone --nproc_per_node=<gpu-count>`，将 prompt 分配到多个 GPU。

## 训练

选择已有的 `libero_easywam_<architecture>_cosmos25` task；LoRA 使用 `_lora` 后缀。例如：

```bash
# LIBERO 上的全参数 MoT-Joint 训练
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_joint_cosmos25

# LIBERO 上的 LoRA Hidden 训练
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_hidden_cosmos25_lora
```

架构名称为 `unified`、`mot`、`mot_joint`、`mot_idm` 和 `hidden`。训练结果写入 `output_dir`，默认路径为 `runs/train/<date>_<time>/`。

模型使用 `standard` ActionDiT 预设。视觉注意力由 `model.backbone.video_attention_mask_mode` 控制，可选 `first_frame_causal`（默认）、`per_frame_causal` 或 `bidirectional`。`per_frame_causal` 模式下，同一帧内的 token 双向交互，每一帧可见自身和此前的帧。

## 评测

使用与 checkpoint 一致的 task，以匹配模型维度和 backbone 设置：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_easywam_mot_cosmos25 \
  ckpt=<path/to/checkpoint.pt>
```

评测自动加载 Cosmos-Reason1-7B。仿真环境、归一化统计、批量推理和结果目录见 [LIBERO](../benchmark/libero_zh.md) 或 [LIBERO-Plus](../benchmark/libero_plus_zh.md) 评测指南。

## 独立视觉训练

`model.backbone.cosmos25` 导出 `Cosmos25Core`。通过 `Cosmos25Core.from_cosmos25_pretrained(...)` 加载权重，调用 `model(sample)` 计算视觉损失，使用 `infer`、`save_checkpoint` 和 `load_checkpoint` 进行生成与恢复。样本包含归一化的 `video`（`[B,3,T,H,W]`），以及 `prompt` 或缓存的 `context`/`context_mask`；使用缓存时请移除 `prompt`。帧数和图像尺寸须满足上述 VAE 约束。无需动作和状态字段。仅优化 `model.dit` 时，应冻结 VAE 和文本编码器参数。

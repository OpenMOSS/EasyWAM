# Wan2.2-TI2V-5B Backbone

[English](wan22.md) | [Backbone 索引](README_zh.md) | [模型配置](../config/models_zh.md) | [返回项目 README](../../../README_zh.md)

Wan2.2 支持 EasyWAM-Unified、MoT、MoT-Joint、MoT-IDM 和 Hidden，并提供全参数与 LoRA 模型配置。默认路径和运行设置位于 `configs/model/backbone/wan22.yaml`。

## 准备模型权重

安装 EasyWAM 和 Hugging Face CLI 后，在项目根目录执行：

```bash
mkdir -p checkpoints
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B
```

该目录提供 Video DiT、VAE、UMT5 编码器和 tokenizer。默认配置从 `checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl` 加载 UMT5 文件。

## 初始化 ActionDiT

MoT、MoT-Joint、MoT-IDM 和 Hidden 使用独立的 ActionDiT。生成一次插值初始化权重：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot_wan22.yaml \
  --backbone wan22 \
  --output checkpoints/ActionDiT_Wan22_5B_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified 不使用该文件。自定义权重位置时，可覆盖 `model.backbone.model_id`、`model.backbone.tokenizer_model_id` 或 `model.backbone.action_dit_pretrained_path`。

## 预计算文本特征

准备好 [LIBERO](../data/libero_zh.md) 或 [RoboTwin](../data/robotwin_zh.md) 数据集后，选择对应 task 生成 UMT5 缓存：

```bash
python scripts/precompute_text_embeds.py task=libero_easywam_mot_wan22
python scripts/precompute_text_embeds.py task=robotwin_easywam_mot_wan22
```

Task 决定数据集目录和缓存位置。数据集与 `context_len` 相同时，不同 EasyWAM 架构可复用同一份 Wan2.2 文本缓存。多 GPU 预计算可在脚本前使用 `torchrun --standalone --nproc_per_node=<gpu-count>`。

## 训练

选择已有的 `<benchmark>_easywam_<architecture>_wan22` task。LIBERO 还提供带 `_lora` 后缀的 task。例如：

```bash
# LIBERO 上的全参数 MoT 训练
NPROC_PER_NODE=8 bash scripts/train_zero1.sh \
  task=libero_easywam_mot_wan22

# LIBERO 上的 LoRA Unified 训练
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_unified_wan22_lora
```

架构名称为 `unified`、`mot`、`mot_joint`、`mot_idm` 和 `hidden`。训练结果写入 `output_dir`，默认路径为 `runs/train/<date>_<time>/`。

模型使用 `standard` ActionDiT 预设。视觉注意力由 `model.backbone.video_attention_mask_mode` 控制，可选 `first_frame_causal`（默认）、`per_frame_causal` 或 `bidirectional`。`per_frame_causal` 模式下，同一帧内的 token 双向交互，每一帧可见自身和此前的帧。

## 评测

使用与 checkpoint 一致的 task，以匹配模型维度和 backbone 设置：

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

评测自动加载文本编码器。仿真环境、归一化统计、批量推理和结果目录见 [LIBERO](../benchmark/libero_zh.md)、[LIBERO-Plus](../benchmark/libero_plus_zh.md) 或 [RoboTwin](../benchmark/robotwin_zh.md) 评测指南。

## 独立视觉训练

`model.backbone.wan22` 导出 `Wan22Core`。通过 `Wan22Core.from_wan22_pretrained(...)` 加载权重，调用 `model(sample)` 计算视觉损失，使用 `infer`、`save_checkpoint` 和 `load_checkpoint` 进行生成与恢复。样本包含归一化的 `video`（`[B,3,T,H,W]`）和 `prompt`；帧数满足 `T % 4 == 1`，高宽能被 16 整除。该 Core 在运行时编码 prompt，不读取策略文本缓存；Core 不使用动作和状态字段。仅优化 `model.dit` 时，应冻结 VAE 和文本编码器参数。

# 模型配置

[English](models.md) | [配置索引](README_zh.md) | [返回 README](../../../README_zh.md)

`configs/model/` 中的模型配置选择 EasyWAM 架构、视觉 backbone 和可选的 LoRA adapter。task 配置将模型与数据集、训练参数组合起来。模型命名形式为 `easywam_<architecture>_<backbone>[_lora]`。

## 选择架构

| 架构 | 行为 | 主要设置 |
| --- | --- | --- |
| Unified | 使用共享视觉 Transformer 联合建模视觉和动作 token。 | `action_dim`、`state_dim`、`projector_hidden_dim` |
| MoT | 视觉和动作专家通过混合自注意力交互。 | `action_dit_config`、`action_dit_pretrained_path` |
| MoT-Joint | 两个专家联合去噪未来观测和动作。 | 与 MoT 相同的结构设置 |
| MoT-IDM | 使用 teacher-forcing 未来观测作为动作训练条件。 | `video_cond_noise_prob` |
| Hidden | 使用视觉中间层特征作为 ActionDiT 的条件。 | `video_hidden_layer`、`detach_video_hidden`、`action_dit_config` |

可以选择已有 task，也可以保留其数据与训练设置，覆盖模型组：

```bash
python scripts/train.py --cfg job task=libero_easywam_mot_joint_cosmos25

python scripts/train.py --cfg job \
  task=libero_easywam_mot_wan22 \
  model=easywam_hidden_wan22_lora
```

## 选择 Backbone

| Backbone | Unified | MoT | MoT-Joint | MoT-IDM | Hidden |
| --- | :---: | :---: | :---: | :---: | :---: |
| Wan2.2-TI2V-5B | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cosmos-Predict2.5-2B | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX.2 Klein Base 4B | ✅ | ✅ | ✅ | ✅ | ✅ |

Backbone 配置定义模型路径、文本和注意力维度、scheduler 默认值及 LoRA 目标模块。模型配置选择对应的 Transformer 结构，并从 backbone 获得所需维度。权重下载、文本编码器、视觉预测目标和特定 backbone 的使用约束见 [Backbone 指南](../backbone/README_zh.md)。

训练通常设置 `model.backbone.load_text_encoder=false`，读取缓存的文本特征。文本预计算和 Benchmark 评测在需要编码任务指令时启用文本编码器。

## 从数据配置获得维度

机器人维度由数据 processor 提供：

```yaml
state_dim: ${data.train.processor.proprio_output_dim}
state_position: context
action_dit_config:
  action_dim: ${data.train.processor.action_output_dim}
```

Unified 和 Hidden 还在模型顶层暴露 `action_dim`。接入数据集时，应保持 `shape_meta`、`action_output_dim` 和 `proprio_output_dim` 一致，再通过 `--cfg job` 检查组合配置。

## 配置 ActionDiT

MoT 系列与 Hidden 通过 `model.action_dit_config` 配置动作专家。Unified 将动作投影到共享 Transformer，不使用独立的 ActionDiT。

`preset` 选择默认 block 结构：`standard` 使用顺序堆叠的 Transformer block；`staged` 先使用顺序 block，再使用注意力与 FFN 并行的 block。显式组件设置覆盖预设默认值。

| 配置项 | 取值 | `standard` 默认 | `staged` 默认 |
| --- | --- | --- | --- |
| `cross_attention` | `true`、`false` | true | false |
| `qkv_projection` | `separate`、`fused` | separate | fused |
| `qkv_bias` | `true`、`false` | true | false |
| `qk_norm` | `projection`、`head`、`none` | projection | head |
| `ffn_activation` | `gelu`、`swiglu` | gelu | swiglu |
| `encoder_type` | `mlp`、`linear` | mlp | linear |
| `decoder_type` | `mlp`、`linear`、`time_modulated` | mlp | time_modulated |
| `position_encoding` | `rope_1d`、`rope_nd` | rope_1d | rope_nd |
| `time_embedding_type` | `standard`、`scaled` | standard | scaled |

`num_layers` 控制 standard 层数；`num_layers_double` 和 `num_layers_single` 控制 staged 的两个阶段层数。`hidden_dim` 设置动作宽度，`ffn_dim` 或 `mlp_ratio` 设置 FFN 宽度，`max_action_horizon` 设置最大动作序列长度。staged 并行 block 的融合 QKV 与 FFN 输入共用投影，因此 `qkv_bias` 同时控制该 FFN 输入的 bias。standard block 使用一维 RoPE。

MoT 系列专家需要兼容的 block 拓扑与注意力维度。Hidden 通过交叉注意力读取视觉特征，需要 `cross_attention=true`。调整时应以所选模型配置为基础，并使用与结构匹配的初始化权重。

## 初始化与加载权重

| 设置 | 含义 |
| --- | --- |
| Backbone 路径字段 | 预训练视觉 Transformer、VAE 和文本编码器的位置。 |
| `action_dit_pretrained_path` | 独立 ActionDiT 的可选初始化权重；`null` 表示随机初始化。 |
| `skip_dit_load_from_pretrain` | 加载 EasyWAM checkpoint 前跳过视觉预训练权重。 |
| `ckpt` | Benchmark 评测使用的策略 checkpoint。 |
| `resume` | 恢复训练状态时使用的 checkpoint 或目录。 |

`scripts/preprocess_action_dit_backbone.py` 从视觉 Transformer 权重生成 ActionDiT 初始化。脚本读取 `--model-config` 选择的 backbone，可选的 `--backbone` 用于核对该选择。模型配置和输出路径见对应 backbone 指南。Unified 无需此步骤。加载策略 checkpoint 时，应使用匹配的架构、状态布局和动作结构。

## 全参数训练与 LoRA

选择普通模型配置进行全参数训练，或选择带 `_lora` 后缀的配置对视觉 backbone 使用 LoRA。LoRA 配置组合 `configs/model/lora/video_dit.yaml`；默认 rank 和 alpha 为 `128`，dropout 为零，目标模块由 backbone 指定。

```bash
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_mot_wan22 \
  model=easywam_mot_wan22_lora \
  model.lora.r=64 \
  model.lora.lora_alpha=64
```

文本预计算、训练和评测应使用一致的 task 与模型选择。优化器、checkpoint 保存和恢复训练见[训练指南](training_zh.md)。

## 损失权重与 Scheduler

总目标为 `lambda_video * loss_video + lambda_action * loss_action`。两项权重配置在 `model.loss` 下：

```yaml
loss:
  lambda_video: 1.0
  lambda_action: 1.0
```

默认值由所选模型配置决定。将权重设为 `0` 可关闭对应目标。

`model.backbone.video_scheduler.family` 可选 `continuous_flow` 或 `flow_unipc`，默认值由 backbone 提供。`video_scheduler` 来自 backbone，包含独立动作专家的模型还会定义 `action_scheduler`。`train_shift` 控制训练时间步采样，`infer_shift` 控制推理，`num_train_timesteps` 定义时间步范围。

## 条件与注意力

| 设置 | 用法 |
| --- | --- |
| `state_position` | `context` 将状态加入条件上下文；`sequence` 将状态放在建模序列的动作 token 之后。加载 checkpoint 时保持布局一致。 |
| `video_cond_noise_prob` | MoT-IDM 对 teacher-forcing 观测加噪的概率；范围为 `[0, 1]`，默认 `0.5`。 |
| `video_hidden_layer` | Hidden 读取的视觉 block 零起始索引；默认值来自 backbone。 |
| `detach_video_hidden` | `true` 阻止动作损失通过 Hidden 的视觉特征反向传播；`false` 允许传播。 |
| `projector_hidden_dim` | 状态和动作投影 MLP 的隐藏宽度。 |
| `video_attention_mask_mode` | Backbone 支持的视觉注意力布局；可用模式见其指南。 |

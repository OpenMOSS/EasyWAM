# 模型配置

[English](models.md) | [配置索引](README_zh.md) | [返回 README](../../README_zh.md)

模型 recipe 位于 `configs/model/`，命名形式为 `easywam_<architecture>_<backbone>[_lora]`。task recipe 会选择其中一个模型，并提供模型配置引用的数据维度。

## 选择模型架构

| 架构 | 配置相关行为 | 架构专属设置 |
| --- | --- | --- |
| EasyWAM-Unified | 使用同一个 Video DiT 表示视频、动作和状态 token。 | `action_dim`、`state_dim`、`projector_hidden_dim` |
| EasyWAM-MoT | 独立的 Video DiT 和 Action DiT 通过混合自注意力交互。 | `action_dit_config`、`action_dit_pretrained_path` |
| EasyWAM-MoT-Joint | 两个 expert 联合去噪未来视频和动作 token。 | 与 MoT 相同的结构设置 |
| EasyWAM-MoT-IDM | 增加 teacher-forcing 条件视频分支，用于动作预测。 | `[0, 1]` 范围内的 `video_cond_noise_prob` |
| EasyWAM-Hidden | 使用 Video DiT 中间层特征作为 Action DiT 条件。 | `video_hidden_layer`、`detach_video_hidden` |

应选择对应 task，而不是手工修改 `_target_`：

```bash
# LIBERO 上的 MoT-Joint + Cosmos2.5
python scripts/train.py --cfg job task=libero_easywam_mot_joint_cosmos25

# RoboTwin 上使用 LoRA 的 Hidden + Wan2.2
python scripts/train.py --cfg job task=robotwin_easywam_hidden_wan22_lora
```

## Backbone 支持

| Backbone | Unified | MoT | MoT-Joint | MoT-IDM | Hidden |
| --- | :---: | :---: | :---: | :---: | :---: |
| Wan2.2-TI2V-5B | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cosmos-Predict2.5-2B | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLUX.2 Klein-4B | — | ✅ | — | — | — |

Backbone 配置包含模型/checkpoint 路径、文本维度、Transformer 维度、scheduler 默认值、Attention 设置和 LoRA 目标模块。Action expert 的层数、head 数和 head 维度必须与 video expert 一致；仓库内 recipe 会从 `model.backbone` 插值得到这些值。

训练配置中的 `model.backbone.load_text_encoder` 为 `false`，因为数据集直接读取预计算 embedding；评测配置将其设为 `true`，使 policy 能够编码运行时任务指令。

FLUX.2 的源码、checkpoint 迁移和 Qwen3 cache 要求请参阅 [FLUX.2 / ImageWAM 接入指南](../instructions/flux2_imagewam_integration.md)。

## 从数据配置获得维度

一般不应在模型 recipe 中硬编码机器人维度：

```yaml
state_dim: ${data.train.processor.proprio_output_dim}
action_dit_config:
  action_dim: ${data.train.processor.action_output_dim}
```

Unified 和 Hidden 在模型顶层暴露 `action_dim`；MoT 系列将其放在 `action_dit_config` 下。接入新数据集时，应同时更新 `shape_meta`、`action_output_dim` 和 `proprio_output_dim`，然后检查完整组合配置。

## Checkpoint 初始化

| 设置 | 含义 |
| --- | --- |
| Backbone 路径字段 | 定位预训练 Video DiT、VAE/tokenizer 和可选文本 encoder。 |
| `action_dit_pretrained_path` | MoT 系列和 Hidden 使用的可选插值 ActionDiT 初始化权重。 |
| `skip_dit_load_from_pretrain` | 跳过预训练 DiT 权重；评测会设为 `true`，随后加载 EasyWAM checkpoint。 |
| `ckpt` | `configs/sim_*.yaml` 定义的评测 checkpoint 路径。 |
| `resume` | Trainer 恢复训练所使用的 checkpoint 或目录。 |

使用根 README 中的 `scripts/preprocess_action_dit_backbone.py` 生成 Wan2.2 或 Cosmos2.5 ActionDiT 初始化权重。EasyWAM-Unified 不使用独立的 ActionDiT checkpoint。

## 全参数训练与 LoRA

`_lora` 模型 recipe 会组合 `configs/model/lora/video_dit.yaml`。默认 adapter 的 rank 和 alpha 均为 `128`，dropout 为零，并使用各 backbone 自己的目标模块。应通过 `_lora` task 选择 LoRA；除非 checkpoint 加载协议明确包含 adapter，否则不要向已有全参数 checkpoint 临时附加 LoRA 组。

可以在组合配置时覆盖 LoRA 参数：

```bash
NPROC_PER_NODE=4 bash scripts/train_zero2.sh \
  task=libero_easywam_mot_wan22_lora \
  model.lora.r=64 \
  model.lora.lora_alpha=64
```

## 损失权重与 Scheduler

所有架构配置均暴露：

```yaml
loss:
  lambda_video: 1.0
  lambda_action: 1.0
```

总目标为 `lambda_video * loss_video + lambda_action * loss_action`。只有在有意关闭对应目标时才将权重设为 `0`；新 recipe 应显式保留两项。FLUX.2 MoT 当前的 `lambda_video` 默认值为 `0.5`。

`video_scheduler` 继承自 backbone。包含独立 ActionDiT 的架构还会定义 `action_scheduler`。`train_shift` 影响训练噪声/时间步采样，`infer_shift` 影响推理，`num_train_timesteps` 定义 scheduler 离散范围。Scheduler 调整会改变模型行为，不应被当作普通速度开关。

## 架构专属设置

- `video_cond_noise_prob` 控制 MoT-IDM 在训练中对 teacher-forcing 条件视频加噪的概率，必须位于 `0` 到 `1` 之间，默认值为 `0.5`。
- `video_hidden_layer` 是 Hidden 使用的 Video DiT block 零起始索引。仓库默认值来自 backbone，且必须小于层数。
- `detach_video_hidden=true` 阻止动作损失通过 Hidden 的视频特征路径反向传播。设为 `false` 会让动作梯度进入该路径，并改变显存占用和优化行为。
- `projector_hidden_dim` 控制较小的状态/动作投影 MLP；修改后 checkpoint shape 将不兼容。
- `video_attention_mask_mode` 属于 backbone 行为设置。仓库内 Wan2.2 和 Cosmos2.5 recipe 使用 `first_frame_causal`；恢复使用其他模式训练的 checkpoint 时不要改变该值。
